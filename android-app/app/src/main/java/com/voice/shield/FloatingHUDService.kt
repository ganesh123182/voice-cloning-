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
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.core.content.ContextCompat
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

    // Explainable AI UI
    private lateinit var llExplainabilityReasons: LinearLayout
    private lateinit var tvReason1: TextView
    private lateinit var tvReason2: TextView
    private lateinit var tvReason3: TextView

    private var pulseAnimator: ValueAnimator? = null
    private var isPulsing = false

    private var hasThreatLatched = false
    private var latchedRiskScore = 0
    private var latchedAdvisory = ""
    private var hasAlertedVibration = false

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
        ContextCompat.registerReceiver(this, hudUpdateReceiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        Log.d(TAG, "FloatingHUDService destroyed")
        try {
            unregisterReceiver(hudUpdateReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister receiver", e)
        }
        
        hasThreatLatched = false
        latchedRiskScore = 0
        latchedAdvisory = ""
        hasAlertedVibration = false
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

            // Explainable AI views
            llExplainabilityReasons = it.findViewById(R.id.ll_explainability_reasons)
            tvReason1 = it.findViewById(R.id.tv_reason_1)
            tvReason2 = it.findViewById(R.id.tv_reason_2)
            tvReason3 = it.findViewById(R.id.tv_reason_3)
            
            btnCloseHud.setOnClickListener {
                // User chose to hide the HUD manually
                stopSelf()
            }
        }

        // Configuration for the Floating Window
        val layoutFlag = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY

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
        @Suppress("ClickableViewAccessibility")
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
            val rawRiskScore = json.optDouble("risk_score", 0.0).toInt()
            val isAlert = json.optBoolean("is_alert", false)
            val serverThreatLatched = json.optBoolean("threat_latched", false)
            val serverLatchedRisk = json.optDouble("latched_risk", 0.0).toInt()
            val callerName = json.optString("caller_name", "Unknown Caller")
            val callerNumber = json.optString("caller_number", "Unknown")
            
            val callerInfo = if (callerName == "Unknown Caller") callerNumber else "$callerName ($callerNumber)"
            
            // Threat Latching State: Remember if an AI threat occurred during this call
            if (isAlert || (speaker == "caller" && rawRiskScore >= 50) || serverThreatLatched) {
                hasThreatLatched = true
                latchedRiskScore = maxOf(latchedRiskScore, rawRiskScore, serverLatchedRisk)
                latchedAdvisory = "Warning: AI voice detected! Do not share OTPs with $callerInfo!"
            }

            // Live score reflects the current active audio window dynamically
            // (drops to 0% when user speaks or pauses, rises when caller speaks)
            val riskScore = rawRiskScore

            // Default colors and text
            var badgeColor = Color.parseColor("#FFCA28") // Amber
            var badgeText = "SCANNING..."
            var meterColor = Color.parseColor("#00E676") // Green
            var advisoryText = "Monitoring audio..."
            var shouldPulse = false

            when (speaker) {
                "user" -> {
                    badgeColor = Color.parseColor("#00E676") // Green
                    badgeText = "👤 USER SPEAKING"
                    meterColor = Color.parseColor("#00E676")
                    advisoryText = if (hasThreatLatched) {
                        "⚠️ You are speaking — Caller was flagged as AI Voice!"
                    } else {
                        "You are speaking (Safe)."
                    }
                    stopPulseAnimation()
                }
                "caller" -> {
                    if (isAlert || riskScore >= 50) {
                        badgeColor = Color.parseColor("#FF1744") // Red
                        val label = json.optString("label", "FAKE VOICE DETECTED")
                        badgeText = if (label.contains("SCAM", ignoreCase = true)) "🚨 SCAM CALL DETECTED" else "🚨 FAKE VOICE DETECTED"
                        meterColor = Color.parseColor("#FF1744")
                        advisoryText = if (riskScore >= 100) {
                            "CRITICAL: Hang up immediately! Voice clone scam."
                        } else {
                            "Warning: AI voice detected! Do not share OTPs with $callerInfo!"
                        }
                        shouldPulse = true

                        // Distinct haptic vibration alert fires ONCE per threat event
                        if (!hasAlertedVibration) {
                            hasAlertedVibration = true
                            triggerSingleVibrationAlert()
                        }
                    } else if (riskScore > 30) {
                        badgeColor = Color.parseColor("#FF9100") // Orange
                        badgeText = "SUSPICIOUS AUDIO"
                        meterColor = Color.parseColor("#FF9100")
                        advisoryText = "Unusual voice patterns from $callerInfo."
                        stopPulseAnimation()
                    } else {
                        badgeColor = Color.parseColor("#00E676") // Green
                        badgeText = "VERIFIED CALLER"
                        meterColor = Color.parseColor("#00E676")
                        advisoryText = if (hasThreatLatched) {
                            "Caller speaking — Prior AI activity flagged!"
                        } else {
                            "Voice matches human profile for $callerInfo."
                        }
                        stopPulseAnimation()
                    }
                }
                "silence" -> {
                    meterColor = Color.parseColor("#888888") // Gray
                    if (hasThreatLatched) {
                        badgeColor = Color.parseColor("#FF9100") // Orange
                        badgeText = "⚠️ CALLER PAUSED"
                        advisoryText = "Caller paused — AI voice flagged on this call."
                    } else {
                        badgeColor = Color.parseColor("#888888") // Gray
                        badgeText = "NO SPEECH DETECTED"
                        advisoryText = "Waiting for caller..."
                    }
                    stopPulseAnimation()
                }
                else -> {
                    badgeColor = Color.parseColor("#FFCA28") // Amber
                    badgeText = "SCANNING..."
                    meterColor = Color.parseColor("#FFCA28")
                    advisoryText = "Listening to caller voice..."
                    stopPulseAnimation()
                }
            }

            Log.i(TAG, "updateHUD: speaker=$speaker, risk=$riskScore, badge='$badgeText'")

            // Update Views on Main Thread
            tvStatusBadge.text = badgeText
            tvStatusBadge.setTextColor(badgeColor)
            
            progressRiskMeter.progress = riskScore
            progressRiskMeter.progressTintList = ColorStateList.valueOf(meterColor)
            
            @Suppress("SetTextI18n")
            tvRiskPercentage.text = "$riskScore%"
            tvRiskPercentage.setTextColor(meterColor)
            
            tvThreatAdvisory.text = advisoryText

            // ── Explainable AI Reasons ──
            val reasonsArray = json.optJSONArray("explainability_reasons")
            if (reasonsArray != null && reasonsArray.length() > 0 && (hasThreatLatched || riskScore >= 50)) {
                llExplainabilityReasons.visibility = View.VISIBLE
                val reasonViews = listOf(tvReason1, tvReason2, tvReason3)
                for (i in reasonViews.indices) {
                    if (i < reasonsArray.length()) {
                        reasonViews[i].text = reasonsArray.optString(i, "")
                        reasonViews[i].visibility = View.VISIBLE
                    } else {
                        reasonViews[i].visibility = View.GONE
                    }
                }
            } else if (!hasThreatLatched) {
                llExplainabilityReasons.visibility = View.GONE
                tvReason1.visibility = View.GONE
                tvReason2.visibility = View.GONE
                tvReason3.visibility = View.GONE
            }

            if (shouldPulse) {
                startPulseAnimation()
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

    private fun triggerSingleVibrationAlert() {
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
                    val timings = longArrayOf(0, 250, 150, 250)
                    val amplitudes = intArrayOf(0, VibrationEffect.DEFAULT_AMPLITUDE, 0, VibrationEffect.DEFAULT_AMPLITUDE)
                    vibrator.vibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(longArrayOf(0, 250, 150, 250), -1)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Vibration failed", e)
        }
    }
}
