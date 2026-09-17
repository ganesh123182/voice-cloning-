package com.voice.shield

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController

class FragmentThreatAlert : Fragment(R.layout.fragment_threat_alert) {

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Retrieve arguments passed from navigation
        val riskScore = arguments?.getInt("riskScore", 87) ?: 87
        val label = arguments?.getString("label", "FAKE VOICE DETECTED") ?: "FAKE VOICE DETECTED"

        // Update the score display using the proper ID from XML
        val tvScore = view.findViewById<TextView>(R.id.tv_threat_score)
        tvScore?.text = "$riskScore%"

        // Wire up "Verify Caller" button
        val btnVerify = findButtonByText(view, "Verify Caller")
        btnVerify?.setOnClickListener {
            Toast.makeText(requireContext(), "Re-verifying caller identity...", Toast.LENGTH_SHORT).show()
            // Navigate back to live analysis to re-verify
            try {
                findNavController().navigateUp()
            } catch (_: Exception) {}
        }

        // Wire up "Continue Anyway" button
        val btnContinue = findButtonByText(view, "Continue Anyway")
        btnContinue?.setOnClickListener {
            Toast.makeText(requireContext(), "Proceeding with caution. Stay alert!", Toast.LENGTH_SHORT).show()
            try {
                findNavController().navigateUp()
            } catch (_: Exception) {}
        }
    }

    // Helper to find a Button by its text content since the layout has no IDs on buttons
    private fun findButtonByText(root: View, text: String): Button? {
        if (root is Button && root.text.toString() == text) return root
        if (root is android.view.ViewGroup) {
            for (i in 0 until root.childCount) {
                val found = findButtonByText(root.getChildAt(i), text)
                if (found != null) return found
            }
        }
        return null
    }
}
