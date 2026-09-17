package com.voice.shield

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import org.json.JSONObject

class FragmentLiveAnalysis : Fragment(R.layout.fragment_live_analysis) {

    private var tvSecureBadge: TextView? = null
    private var tvAlertBanner: TextView? = null
    private var tvTranscript: TextView? = null
    private var tvRiskScore: TextView? = null
    private var progRiskScore: ProgressBar? = null
    private var tvSpeakerMatch: TextView? = null
    private var progSpeakerMatch: ProgressBar? = null
    private var btnVerifyIdentity: Button? = null

    private val hudReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val jsonString = intent?.getStringExtra(LiveCallService.EXTRA_JSON_PAYLOAD)
            if (jsonString != null) {
                try {
                    val payload = JSONObject(jsonString)
                    val riskScore = payload.optDouble("risk_score", 0.0).toInt()
                    val label = payload.optString("label", "")
                    val isAlert = payload.optBoolean("is_alert", false)
                    val transcript = payload.optString("transcript", "")
                    val speakerMatch = payload.optInt("speaker_match", 0)
                    val callerName = payload.optString("caller_name", "Unknown Caller")
                    val callerNumber = payload.optString("caller_number", "Unknown")
                    
                    val callerInfo = if (callerName == "Unknown Caller") callerNumber else "$callerName ($callerNumber)"
                    
                    updateAnalysis(riskScore, isAlert, transcript, speakerMatch, callerInfo)

                    if (riskScore > 56) {
                        launchThreatAlert(riskScore, label, callerInfo)
                    }
                } catch (e: Exception) {
                    Log.e("FragmentLiveAnalysis", "Error parsing JSON", e)
                }
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        tvSecureBadge = view.findViewById(R.id.tv_secure_badge)
        tvAlertBanner = view.findViewById(R.id.tv_alert_banner)
        tvTranscript = view.findViewById(R.id.tv_transcript)
        tvRiskScore = view.findViewById(R.id.tv_risk_score)
        progRiskScore = view.findViewById(R.id.prog_risk_score)
        tvSpeakerMatch = view.findViewById(R.id.tv_speaker_match)
        progSpeakerMatch = view.findViewById(R.id.prog_speaker_match)
        btnVerifyIdentity = view.findViewById(R.id.btn_verify_identity)

        // Wire "Verify Caller Identity" button — launch Simulate call
        btnVerifyIdentity?.setOnClickListener {
            val intent = Intent(requireContext(), MockCallActivity::class.java)
            startActivity(intent)
        }
    }

    override fun onResume() {
        super.onResume()
        val filter = IntentFilter(LiveCallService.ACTION_HUD_UPDATE)
        ContextCompat.registerReceiver(
            requireContext(),
            hudReceiver,
            filter,
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
    }

    override fun onPause() {
        super.onPause()
        try {
            requireContext().unregisterReceiver(hudReceiver)
        } catch (_: Exception) {
            // Receiver may not have been registered
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        tvSecureBadge = null
        tvAlertBanner = null
        tvTranscript = null
        tvRiskScore = null
        progRiskScore = null
        tvSpeakerMatch = null
        progSpeakerMatch = null
        btnVerifyIdentity = null
    }

    private fun updateAnalysis(riskScore: Int, isSuspicious: Boolean, transcript: String, speakerMatch: Int, callerInfo: String) {
        if (!isAdded || view == null) return

        activity?.runOnUiThread {
            tvRiskScore?.text = "$riskScore%"
            progRiskScore?.progress = riskScore
            
            tvSpeakerMatch?.text = "$speakerMatch%"
            progSpeakerMatch?.progress = speakerMatch
            
            if (transcript.isNotEmpty()) {
                tvTranscript?.text = "Transcript: \"$transcript\""
            }

            if (isSuspicious || riskScore > 45) {
                tvAlertBanner?.visibility = View.VISIBLE
                tvAlertBanner?.text = "Suspicious Audio from $callerInfo"
                tvSecureBadge?.text = "THREAT DETECTED"
                tvSecureBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_suspicious))
                tvRiskScore?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_suspicious))
                progRiskScore?.progressTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_suspicious)
            } else {
                tvAlertBanner?.visibility = View.GONE
                tvSecureBadge?.text = "SECURE"
                tvSecureBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_safe))
                tvRiskScore?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_safe))
                progRiskScore?.progressTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_safe)
            }
        }
    }

    private fun launchThreatAlert(riskScore: Int, label: String, callerInfo: String) {
        if (!isAdded || view == null) return
        
        try {
            val bundle = Bundle().apply {
                putInt("riskScore", riskScore)
                putString("label", label)
                putString("callerInfo", callerInfo)
            }
            findNavController().navigate(R.id.action_liveAnalysis_to_threatAlert, bundle)
        } catch (e: Exception) {
            Log.e("FragmentLiveAnalysis", "Navigation failed", e)
        }
    }
}
