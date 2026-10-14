package com.malikh.manualcam

import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.media.ImageReader
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Range
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.roundToLong

class MainActivity : Activity() {

    private lateinit var previewView: TextureView
    private lateinit var aeSwitch: Switch
    private lateinit var afSwitch: Switch
    private lateinit var readout: TextView
    private lateinit var isoLabel: TextView
    private lateinit var shutterLabel: TextView
    private lateinit var focusLabel: TextView
    private lateinit var zoomLabel: TextView
    private lateinit var isoSlider: SeekBar
    private lateinit var shutterSlider: SeekBar
    private lateinit var focusSlider: SeekBar
    private lateinit var zoomSlider: SeekBar
    private lateinit var actionBtn: Button
    private lateinit var modeGroup: android.widget.RadioGroup

    private lateinit var cameraManager: CameraManager
    private var cameraId: String = ""
    private var cameraDevice: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null

    private var bgThread: HandlerThread? = null
    private var bgHandler: Handler? = null

    private var sensorOrientation = 90
    private lateinit var isoRange: Range<Int>
    private lateinit var exposureRange: Range<Long>

    // Focus, in diopters (1/meters): 0 = infinity, minimumFocusDistance = closest
    private var minFocusDiopters = 10f
    private var manualFocusDiopters = 0f

    private lateinit var zoomRange: Range<Float>
    private var zoomRatio = 1f

    // Physical device rotation (0/90/180/270), tracked even though the UI is portrait-locked,
    // so captures are upright when the phone is mounted in landscape.
    private var deviceOrientation = 0
    private var orientationListener: android.view.OrientationEventListener? = null

    // Manual values (defaults get clamped to the device range on open)
    private var manualIso = 100
    private var manualExposureNs = 10_000_000L // 10 ms

    // Video runs at 30 fps: a frame cannot expose longer than the frame interval
    private val VIDEO_FPS = 30
    private val videoMaxExposureNs = 1_000_000_000L / VIDEO_FPS
    private var videoMode = false

    // Shutter range for the current mode
    private val effExposureRange: Range<Long>
        get() = if (videoMode)
            Range(exposureRange.lower, minOf(exposureRange.upper, videoMaxExposureNs))
        else exposureRange

    // Video state
    private var recorder: MediaRecorder? = null
    private var recording = false
    private var videoUri: Uri? = null
    private var videoPfd: ParcelFileDescriptor? = null

    private val aeOn get() = aeSwitch.isChecked
    private val afOn get() = afSwitch.isChecked

    // Remote control for scripted sweeps:
    //   adb shell am broadcast -a com.malikh.manualcam.CTRL \
    //     [--ez ae true|false] [--ei iso N] [--el shutter_ns N] [--ez af true|false] \
    //     [--ef focus_diopters F] [--ef zoom F] [--es mode photo|video] \
    //     [--es record start|stop] [--ez photo true]
    private val ctrlReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(c: Context?, intent: android.content.Intent?) {
            intent ?: return
            runOnUiThread { handleCtrl(intent) }
        }
    }

    private fun handleCtrl(i: android.content.Intent) {
        i.getStringExtra("mode")?.let {
            modeGroup.check(if (it == "video") R.id.modeVideo else R.id.modePhoto)
        }
        if (i.hasExtra("ae")) aeSwitch.isChecked = i.getBooleanExtra("ae", true)
        if (i.hasExtra("af")) afSwitch.isChecked = i.getBooleanExtra("af", true)
        if (i.hasExtra("iso")) {
            manualIso = i.getIntExtra("iso", manualIso).coerceIn(isoRange.lower, isoRange.upper)
            isoSlider.progress = sliderFromIso(manualIso)
        }
        if (i.hasExtra("shutter_ns")) {
            manualExposureNs = i.getLongExtra("shutter_ns", manualExposureNs)
                .coerceIn(effExposureRange.lower, effExposureRange.upper)
            shutterSlider.progress = sliderFromExposure(manualExposureNs)
        }
        if (i.hasExtra("zoom") && ::zoomRange.isInitialized) {
            zoomRatio = i.getFloatExtra("zoom", zoomRatio)
                .coerceIn(zoomRange.lower, zoomRange.upper)
            val lo = ln(zoomRange.lower.toDouble()); val hi = ln(zoomRange.upper.toDouble())
            if (hi > lo) zoomSlider.progress =
                ((ln(zoomRatio.toDouble()) - lo) / (hi - lo) * 1000).roundToInt().coerceIn(0, 1000)
        }
        if (i.hasExtra("focus_diopters")) {
            manualFocusDiopters = i.getFloatExtra("focus_diopters", manualFocusDiopters)
                .coerceIn(0f, minFocusDiopters)
            focusSlider.progress = (manualFocusDiopters / minFocusDiopters * 1000).roundToInt()
        }
        when (i.getStringExtra("record")) {
            "start" -> if (videoMode && !recording) { startRecording(); return }
            "stop" -> if (recording) { stopRecording(); return }
        }
        if (i.getBooleanExtra("photo", false)) { takePhoto(); return }
        updateManualLabels()
        updateRepeating()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.preview)
        // Fit the preview to 16:9 inside its container so it shows the true capture
        // frame undistorted (the stream is 1920x1080)
        val container = findViewById<android.widget.FrameLayout>(R.id.previewContainer)
        container.post {
            var w = container.width; var h = w * 9 / 16
            if (h > container.height) { h = container.height; w = h * 16 / 9 }
            previewView.layoutParams = android.widget.FrameLayout.LayoutParams(
                w, h, android.view.Gravity.CENTER)
            previewView.post { configureTransform(w, h) }
        }
        aeSwitch = findViewById(R.id.aeSwitch)
        afSwitch = findViewById(R.id.afSwitch)
        readout = findViewById(R.id.readout)
        isoLabel = findViewById(R.id.isoLabel)
        shutterLabel = findViewById(R.id.shutterLabel)
        focusLabel = findViewById(R.id.focusLabel)
        focusSlider = findViewById(R.id.focusSlider)
        zoomLabel = findViewById(R.id.zoomLabel)
        zoomSlider = findViewById(R.id.zoomSlider)
        isoSlider = findViewById(R.id.isoSlider)
        shutterSlider = findViewById(R.id.shutterSlider)
        actionBtn = findViewById(R.id.actionBtn)
        modeGroup = findViewById(R.id.modeGroup)

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        aeSwitch.setOnCheckedChangeListener { _, _ ->
            updateSliderEnabledState()
            updateRepeating()
        }
        afSwitch.setOnCheckedChangeListener { _, _ ->
            updateSliderEnabledState()
            updateRepeating()
        }
        focusSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                manualFocusDiopters = minFocusDiopters * progress / 1000f
                focusLabel.text = "Focus  ${formatFocus(manualFocusDiopters)}"
                if (fromUser && !afOn) updateRepeating()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
        zoomSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!::zoomRange.isInitialized) return
                val lo = ln(zoomRange.lower.toDouble()); val hi = ln(zoomRange.upper.toDouble())
                zoomRatio = exp(lo + (hi - lo) * progress / 1000.0).toFloat()
                    .coerceIn(zoomRange.lower, zoomRange.upper)
                zoomLabel.text = String.format(Locale.US, "Zoom  %.1fx", zoomRatio)
                if (fromUser) updateRepeating()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        orientationListener = object : android.view.OrientationEventListener(this) {
            override fun onOrientationChanged(o: Int) {
                if (o == ORIENTATION_UNKNOWN) return
                deviceOrientation = ((o + 45) / 90 * 90) % 360
            }
        }

        val filter = android.content.IntentFilter("com.malikh.manualcam.CTRL")
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            registerReceiver(ctrlReceiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(ctrlReceiver, filter)
        }
        val sliderListener = object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                manualIso = isoFromSlider(isoSlider.progress)
                manualExposureNs = exposureFromSlider(shutterSlider.progress)
                updateManualLabels()
                if (fromUser && !aeOn) updateRepeating()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        }
        isoSlider.setOnSeekBarChangeListener(sliderListener)
        shutterSlider.setOnSeekBarChangeListener(sliderListener)

        modeGroup.setOnCheckedChangeListener { _, checkedId ->
            videoMode = checkedId == R.id.modeVideo
            // Re-clamp the manual shutter to the new mode's range
            manualExposureNs = manualExposureNs.coerceIn(effExposureRange.lower, effExposureRange.upper)
            shutterSlider.progress = sliderFromExposure(manualExposureNs)
            updateManualLabels()
            actionBtn.text = if (videoMode) "Record" else "Take photo"
            updateRepeating()
        }
        actionBtn.setOnClickListener {
            if (videoMode) { if (recording) stopRecording() else startRecording() }
            else takePhoto()
        }

        updateSliderEnabledState()

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO), 1)
        }
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (results.isNotEmpty() && results[0] == PackageManager.PERMISSION_GRANTED) {
            if (previewView.isAvailable) openCamera()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        orientationListener?.enable()
        startBgThread()
        if (previewView.isAvailable) {
            if (hasPermissions()) openCamera()
        } else {
            previewView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
                    if (hasPermissions()) openCamera()
                }
                override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
                override fun onSurfaceTextureDestroyed(st: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
            }
        }
    }

    override fun onPause() {
        orientationListener?.disable()
        if (recording) stopRecording()
        closeCamera()
        stopBgThread()
        super.onPause()
    }

    override fun onDestroy() {
        unregisterReceiver(ctrlReceiver)
        super.onDestroy()
    }

    private fun hasPermissions() =
        checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startBgThread() {
        bgThread = HandlerThread("camera").also { it.start() }
        bgHandler = Handler(bgThread!!.looper)
    }

    private fun stopBgThread() {
        bgThread?.quitSafely()
        bgThread?.join()
        bgThread = null
        bgHandler = null
    }

    // ---------- Camera setup ----------

    private fun openCamera() {
        cameraId = cameraManager.cameraIdList.first { id ->
            cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }
        val chars = cameraManager.getCameraCharacteristics(cameraId)
        sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE) ?: Range(100, 3200)
        exposureRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            ?: Range(100_000L, 100_000_000L)
        // The standard key reports a 1/10 s ceiling, but the S21 FE HAL accepts and delivers
        // longer exposures (verified: a requested 1.0 s frame came back as exactly 1.0 s in
        // CaptureResult). Samsung's vendor tag advertises 30 s. Extend photo mode to 1 s.
        exposureRange = Range(exposureRange.lower, maxOf(exposureRange.upper, 1_000_000_000L))

        minFocusDiopters = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 10f
        zoomRange = chars.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) ?: Range(1f, 1f)
        zoomRatio = zoomRatio.coerceIn(zoomRange.lower, zoomRange.upper)

        manualIso = manualIso.coerceIn(isoRange.lower, isoRange.upper)
        manualExposureNs = manualExposureNs.coerceIn(effExposureRange.lower, effExposureRange.upper)
        manualFocusDiopters = manualFocusDiopters.coerceIn(0f, minFocusDiopters)
        runOnUiThread {
            isoSlider.progress = sliderFromIso(manualIso)
            shutterSlider.progress = sliderFromExposure(manualExposureNs)
            focusSlider.progress = (manualFocusDiopters / minFocusDiopters * 1000).roundToInt()
            val zlo = ln(zoomRange.lower.toDouble()); val zhi = ln(zoomRange.upper.toDouble())
            zoomSlider.progress = if (zhi > zlo)
                ((ln(zoomRatio.toDouble()) - zlo) / (zhi - zlo) * 1000).roundToInt().coerceIn(0, 1000)
            else 0
            updateManualLabels()
        }

        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        // Prefer the largest 16:9 JPEG so photo FOV matches the preview and video streams
        val jpegSizes = map.getOutputSizes(android.graphics.ImageFormat.JPEG)
        val jpegSize = jpegSizes
            .filter { kotlin.math.abs(it.width.toDouble() / it.height - 16.0 / 9.0) < 0.02 }
            .maxByOrNull { it.width.toLong() * it.height }
            ?: jpegSizes.maxByOrNull { it.width.toLong() * it.height }!!
        imageReader = ImageReader.newInstance(jpegSize.width, jpegSize.height, android.graphics.ImageFormat.JPEG, 2)
        imageReader!!.setOnImageAvailableListener({ reader ->
            reader.acquireNextImage().use { image ->
                val buf = image.planes[0].buffer
                val bytes = ByteArray(buf.remaining()); buf.get(bytes)
                savePhoto(bytes)
            }
        }, bgHandler)

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                startPreviewSession()
            }
            override fun onDisconnected(device: CameraDevice) { device.close(); cameraDevice = null }
            override fun onError(device: CameraDevice, error: Int) {
                device.close(); cameraDevice = null
                runOnUiThread { Toast.makeText(this@MainActivity, "Camera error $error", Toast.LENGTH_LONG).show() }
            }
        }, bgHandler)
    }

    private fun closeCamera() {
        session?.close(); session = null
        cameraDevice?.close(); cameraDevice = null
        imageReader?.close(); imageReader = null
    }

    private fun previewSurface(): Surface {
        val texture = previewView.surfaceTexture!!
        texture.setDefaultBufferSize(1920, 1080)
        return Surface(texture)
    }

    private fun startPreviewSession() {
        val device = cameraDevice ?: return
        val surface = previewSurface()
        @Suppress("DEPRECATION")
        device.createCaptureSession(listOf(surface, imageReader!!.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                    session = s
                    updateRepeating()
                }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    runOnUiThread { Toast.makeText(this@MainActivity, "Session config failed", Toast.LENGTH_LONG).show() }
                }
            }, bgHandler)
    }

    // ---------- Exposure control ----------

    private fun applyExposure(builder: CaptureRequest.Builder) {
        if (aeOn) {
            builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        } else {
            builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, manualExposureNs)
            builder.set(CaptureRequest.SENSOR_SENSITIVITY, manualIso)
        }
        if (::zoomRange.isInitialized && zoomRange.upper > zoomRange.lower) {
            builder.set(CaptureRequest.CONTROL_ZOOM_RATIO, zoomRatio)
        }
        if (afOn) {
            builder.set(
                CaptureRequest.CONTROL_AF_MODE,
                if (videoMode) CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                else CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            )
        } else {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualFocusDiopters)
        }
    }

    private val previewCallback = object : CameraCaptureSession.CaptureCallback() {
        private var frame = 0
        override fun onCaptureCompleted(s: CameraCaptureSession, req: CaptureRequest, result: TotalCaptureResult) {
            if (frame++ % 10 != 0) return
            val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY)
            if (exp != null && iso != null) {
                if (!aeOn) android.util.Log.d("ManualCam",
                    "requested=${manualExposureNs}ns actual=${exp}ns iso_req=$manualIso iso_actual=$iso")
                val focus = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
                val focusStr = if (focus != null) "  f ${formatFocus(focus)}" else ""
                runOnUiThread { readout.text = "actual: ${formatExposure(exp)}  ISO $iso$focusStr" }
            }
        }
    }

    private fun updateRepeating() {
        val device = cameraDevice ?: return
        val s = session ?: return
        try {
            val template = if (recording) CameraDevice.TEMPLATE_RECORD else CameraDevice.TEMPLATE_PREVIEW
            val builder = device.createCaptureRequest(template)
            builder.addTarget(previewSurface())
            if (recording) recorder?.surface?.let { builder.addTarget(it) }
            applyExposure(builder)
            s.setRepeatingRequest(builder.build(), previewCallback, bgHandler)
        } catch (e: Exception) {
            // Session may be mid-teardown; ignore
        }
    }

    private fun updateSliderEnabledState() {
        isoSlider.isEnabled = !aeOn
        shutterSlider.isEnabled = !aeOn
        val alpha = if (aeOn) 0.4f else 1.0f
        isoSlider.alpha = alpha; shutterSlider.alpha = alpha
        isoLabel.alpha = alpha; shutterLabel.alpha = alpha

        focusSlider.isEnabled = !afOn
        val fAlpha = if (afOn) 0.4f else 1.0f
        focusSlider.alpha = fAlpha; focusLabel.alpha = fAlpha
    }

    // Upright-capture rotation for the back camera. The activity is sensorLandscape, so
    // display rotation tracks how the phone is mounted (more reliable than the raw
    // orientation sensor, which can misreport in a fixed rig).
    private fun captureOrientation(): Int {
        @Suppress("DEPRECATION")
        val deg = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (sensorOrientation - deg + 360) % 360
    }

    // TextureView streams are only auto-oriented for portrait displays; in a landscape
    // activity we must counter-rotate the buffer ourselves (Camera2Basic pattern).
    private fun configureTransform(viewWidth: Int, viewHeight: Int) {
        @Suppress("DEPRECATION")
        val rotation = windowManager.defaultDisplay.rotation
        val matrix = android.graphics.Matrix()
        val viewRect = android.graphics.RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
        val bufferRect = android.graphics.RectF(0f, 0f, 1080f, 1920f)
        val centerX = viewRect.centerX(); val centerY = viewRect.centerY()
        if (rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
            matrix.setRectToRect(viewRect, bufferRect, android.graphics.Matrix.ScaleToFit.FILL)
            val scale = maxOf(viewHeight.toFloat() / 1080f, viewWidth.toFloat() / 1920f)
            matrix.postScale(scale, scale, centerX, centerY)
            matrix.postRotate((90 * (rotation - 2)).toFloat(), centerX, centerY)
        } else if (rotation == Surface.ROTATION_180) {
            matrix.postRotate(180f, centerX, centerY)
        }
        previewView.setTransform(matrix)
    }

    // Diopters -> human distance: 0 = infinity, else 1/d meters
    private fun formatFocus(diopters: Float): String {
        if (diopters <= 0.01f) return "∞"
        val m = 1f / diopters
        return if (m >= 1f) String.format(Locale.US, "%.1f m", m)
        else String.format(Locale.US, "%.0f cm", m * 100)
    }

    // Slider (0..1000) <-> value mappings, log scale
    private fun isoFromSlider(p: Int): Int {
        val lo = ln(isoRange.lower.toDouble()); val hi = ln(isoRange.upper.toDouble())
        return exp(lo + (hi - lo) * p / 1000.0).roundToInt().coerceIn(isoRange.lower, isoRange.upper)
    }
    private fun sliderFromIso(iso: Int): Int {
        val lo = ln(isoRange.lower.toDouble()); val hi = ln(isoRange.upper.toDouble())
        return ((ln(iso.toDouble()) - lo) / (hi - lo) * 1000).roundToInt().coerceIn(0, 1000)
    }
    private fun exposureFromSlider(p: Int): Long {
        val r = effExposureRange
        val lo = ln(r.lower.toDouble()); val hi = ln(r.upper.toDouble())
        return exp(lo + (hi - lo) * p / 1000.0).roundToLong().coerceIn(r.lower, r.upper)
    }
    private fun sliderFromExposure(ns: Long): Int {
        val r = effExposureRange
        val lo = ln(r.lower.toDouble()); val hi = ln(r.upper.toDouble())
        return ((ln(ns.toDouble()) - lo) / (hi - lo) * 1000).roundToInt().coerceIn(0, 1000)
    }

    // Native-camera style: 1/n for sub-second, plain seconds above that
    private fun formatExposure(ns: Long): String {
        val sec = ns / 1e9
        return if (sec >= 1.0) String.format(Locale.US, "%.1f s", sec)
        else String.format(Locale.US, "1/%d", (1.0 / sec).roundToInt())
    }

    private fun updateManualLabels() {
        isoLabel.text = "ISO  $manualIso"
        shutterLabel.text = "Shutter  ${formatExposure(manualExposureNs)}"
        focusLabel.text = "Focus  ${formatFocus(manualFocusDiopters)}"
        zoomLabel.text = String.format(Locale.US, "Zoom  %.1fx", zoomRatio)
    }

    // ---------- Photo ----------

    private fun takePhoto() {
        val device = cameraDevice ?: return
        val s = session ?: return
        if (recording) { Toast.makeText(this, "Stop recording first", Toast.LENGTH_SHORT).show(); return }
        val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
        builder.addTarget(imageReader!!.surface)
        applyExposure(builder)
        builder.set(CaptureRequest.JPEG_ORIENTATION, captureOrientation())
        s.capture(builder.build(), null, bgHandler)
    }

    private fun savePhoto(bytes: ByteArray) {
        val name = "MC_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/ManualCam")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri != null) {
            contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
            runOnUiThread { Toast.makeText(this, "Saved $name", Toast.LENGTH_SHORT).show() }
        }
    }

    // ---------- Video ----------

    private fun startRecording() {
        val device = cameraDevice ?: return
        val name = "MC_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".mp4"
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "DCIM/ManualCam")
        }
        videoUri = contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: return
        videoPfd = contentResolver.openFileDescriptor(videoUri!!, "w")

        recorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setOutputFile(videoPfd!!.fileDescriptor)
            setVideoEncodingBitRate(20_000_000)
            setVideoFrameRate(30)
            setVideoSize(1920, 1080)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            setOrientationHint(captureOrientation())
            prepare()
        }

        session?.close(); session = null
        val surfaces = listOf(previewSurface(), recorder!!.surface)
        @Suppress("DEPRECATION")
        device.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                session = s
                recording = true
                recorder!!.start()
                updateRepeating()
                runOnUiThread {
                    actionBtn.text = "Stop"
                    for (i in 0 until modeGroup.childCount) modeGroup.getChildAt(i).isEnabled = false
                }
            }
            override fun onConfigureFailed(s: CameraCaptureSession) {
                cleanupRecorder(abandon = true)
                runOnUiThread { Toast.makeText(this@MainActivity, "Record session failed", Toast.LENGTH_LONG).show() }
            }
        }, bgHandler)
    }

    private fun stopRecording() {
        recording = false
        try { recorder?.stop() } catch (e: Exception) { /* stopped too soon */ }
        cleanupRecorder(abandon = false)
        runOnUiThread {
            actionBtn.text = "Record"
            for (i in 0 until modeGroup.childCount) modeGroup.getChildAt(i).isEnabled = true
        }
        session?.close(); session = null
        if (cameraDevice != null) startPreviewSession()
    }

    private fun cleanupRecorder(abandon: Boolean) {
        recorder?.release(); recorder = null
        videoPfd?.close(); videoPfd = null
        if (abandon && videoUri != null) contentResolver.delete(videoUri!!, null, null)
        else if (videoUri != null) runOnUiThread { Toast.makeText(this, "Video saved", Toast.LENGTH_SHORT).show() }
        videoUri = null
    }
}
