package com.voice.shield

import android.animation.ArgbEvaluator
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import org.json.JSONObject

/**
 * Service responsible for displaying the Floating Heads-Up Display (HUD)
 * over other apps during an active phone call.
 */
class FloatingHUDService : Service() {

    companion object {
        private const val TAG = "FloatingHUDService"
    }

    private lateinit var windowManager: WindowManager
    private var floatingView: View? = null
    private lateinit var layoutParams: WindowManager.LayoutParams

    // UI References
    private lateinit var tvStatusBadge: TextView
    private lateinit var progressRiskMeter: ProgressBar
    private lateinit var tvRiskPercentage: TextView
    private lateinit var tvThreatAdvisory: TextView
    private lateinit var rootContainer: View
    private lateinit var btnCloseHud: ImageButton

    private var pulseAnimator: ValueAnimator? = null
    private var isPulsing = false

    // Broadcast receiver to get updates from LiveCallService
    private val hudUpdateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == LiveCallService.ACTION_HUD_UPDATE) {
                val jsonPayload = intent.getStringExtra(LiveCallService.EXTRA_JSON_PAYLOAD)
                jsonPayload?.let { updateHUD(it) }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "FloatingHUDService created")

        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        inflateAndAttachHUD()
        setupDragListener()

        // Register receiver for local broadcasts from LiveCallService
        val filter = IntentFilter(LiveCallService.ACTION_HUD_UPDATE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(hudUpdateReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(hudUpdateReceiver, filter)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        Log.d(TAG, "FloatingHUDService destroyed")
        try {
            unregisterReceiver(hudUpdateReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister receiver", e)
        }
        
        pulseAnimator?.cancel()
        
        floatingView?.let {
            if (it.isAttachedToWindow) {
                windowManager.removeView(it)
            }
        }
        super.onDestroy()
    }

    private fun inflateAndAttachHUD() {
        val inflater = getSystemService(LAYOUT_INFLATER_SERVICE) as LayoutInflater
        floatingView = inflater.inflate(R.layout.layout_floating_hud, null)

        // Map UI components
        floatingView?.let {
            tvStatusBadge = it.findViewById(R.id.tv_status_badge)
            progressRiskMeter = it.findViewById(R.id.progress_risk_meter)
            tvRiskPercentage = it.findViewById(R.id.tv_risk_percentage)
            tvThreatAdvisory = it.findViewById(R.id.tv_threat_advisory)
            rootContainer = it.findViewById(R.id.hud_root_container)
            btnCloseHud = it.findViewById(R.id.btn_close_hud)
            
            btnCloseHud.setOnClickListener {
                // User chose to hide the HUD manually
                stopSelf()
            }
        }

        // Configuration for the Floating Window
        val layoutFlag = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            layoutFlag,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            y = 100 // initial vertical offset
        }

        try {
            windowManager.addView(floatingView, layoutParams)
        } catch (e: WindowManager.BadTokenException) {
            Log.e(TAG, "SYSTEM_ALERT_WINDOW permission likely denied", e)
        }
    }

    private fun setupDragListener() {
        floatingView?.setOnTouchListener(object : View.OnTouchListener {
            private var initialX = 0
            private var initialY = 0
            private var initialTouchX = 0f
            private var initialTouchY = 0f

            override fun onTouch(v: View?, event: MotionEvent): Boolean {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = layoutParams.x
                        initialY = layoutParams.y
                        initialTouchX = event.rawX
                        initialTouchY = event.rawY
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        layoutParams.x = initialX + (event.rawX - initialTouchX).toInt()
                        layoutParams.y = initialY + (event.rawY - initialTouchY).toInt()
                        windowManager.updateViewLayout(floatingView, layoutParams)
                        return true
                    }
                }
                return false
            }
        })
    }

    private fun updateHUD(jsonText: String) {
        try {
            val json = JSONObject(jsonText)
            val speaker = json.optString("speaker", "unknown")
            val riskScore = json.optDouble("risk_score", 0.0).toInt()
            
            // Default colors
            var badgeColor = Color.parseColor("#FFCA28") // Amber
            var badgeText = "SCANNING..."
            var meterColor = Color.parseColor("#00E676") // Green
            var advisoryText = "Monitoring audio..."
            var shouldPulse = false

            when (speaker) {
                "user" -> {
                    badgeColor = Color.parseColor("#00E676") // Green
                    badgeText = "VERIFIED USER"
                    advisoryText = "You are speaking."
                    meterColor = Color.parseColor("#00E676")
                    stopPulseAnimation()
                }
                "caller" -> {
                    if (riskScore > 70) {
                        badgeColor = Color.parseColor("#FF1744") // Red
                        badgeText = "🚨 FAKE VOICE DETECTED"
                        advisoryText = "Warning: Do not share OTPs or transfer money!"
                        meterColor = Color.parseColor("#FF1744")
                        shouldPulse = true
                    } else if (riskScore > 40) {
                        badgeColor = Color.parseColor("#FF9100") // Orange
                        badgeText = "SUSPICIOUS AUDIO"
                        advisoryText = "Unusual voice patterns detected. Stay alert."
                        meterColor = Color.parseColor("#FF9100")
                        stopPulseAnimation()
                    } else {
                        badgeColor = Color.parseColor("#00E676") // Green
                        badgeText = "VERIFIED CALLER"
                        advisoryText = "Voice matches human profile."
                        meterColor = Color.parseColor("#00E676")
                        stopPulseAnimation()
                    }
                }
                "silence" -> {
                    badgeColor = Color.parseColor("#888888") // Gray
                    badgeText = "NO SPEECH DETECTED"
                    advisoryText = "Waiting for caller..."
                    meterColor = Color.parseColor("#888888")
                    stopPulseAnimation()
                }
            }

            // Update Views on Main Thread (Broadcasts run on main thread by default)
            tvStatusBadge.text = badgeText
            tvStatusBadge.setTextColor(badgeColor)
            
            progressRiskMeter.progress = riskScore
            progressRiskMeter.progressTintList = ColorStateList.valueOf(meterColor)
            
            tvRiskPercentage.text = "$riskScore%"
            tvRiskPercentage.setTextColor(meterColor)
            
            tvThreatAdvisory.text = advisoryText

            if (shouldPulse) {
                startPulseAnimation()
                triggerVibrationAlert()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update HUD UI", e)
        }
    }

    private fun startPulseAnimation() {
        if (isPulsing) return
        isPulsing = true
        
        pulseAnimator = ObjectAnimator.ofInt(
            rootContainer,
            "backgroundColor",
            Color.parseColor("#D91A1A1A"), // Dark glass
            Color.parseColor("#D94A0000")  // Dark red glass
        ).apply {
            duration = 500
            setEvaluator(ArgbEvaluator())
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.REVERSE
            start()
        }
    }

    private fun stopPulseAnimation() {
        if (!isPulsing) return
        isPulsing = false
        pulseAnimator?.cancel()
        rootContainer.setBackgroundResource(R.drawable.bg_glassmorphic)
    }

    private fun triggerVibrationAlert() {
        try {
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vibratorManager.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }

            if (vibrator.hasVibrator()) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(500)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Vibration failed", e)
        }
    }
}
