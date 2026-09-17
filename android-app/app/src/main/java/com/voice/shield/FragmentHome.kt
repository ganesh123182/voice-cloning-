package com.voice.shield

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.RecyclerView
import com.voice.shield.api.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class FragmentHome : Fragment(R.layout.fragment_home) {

    private lateinit var adapter: RecentActivityAdapter
    private lateinit var rvActivity: RecyclerView
    private lateinit var pbLoading: ProgressBar

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        rvActivity = view.findViewById(R.id.recycler_recent_activity)
        pbLoading = view.findViewById(R.id.progress_bar)

        val tvUserGreeting = view.findViewById<TextView>(R.id.tv_user_greeting)
        val prefs = requireActivity().getSharedPreferences("TrustVoicePrefs", android.content.Context.MODE_PRIVATE)
        val profileName = prefs.getString("profile_name", "Ganesh") ?: "Ganesh"
        tvUserGreeting?.text = "$profileName 👋"

        // Quick Actions: Voice Enroll
        val btnVoiceEnroll = view.findViewById<LinearLayout>(R.id.btn_voice_enroll)
        btnVoiceEnroll.setOnClickListener {
            findNavController().navigate(R.id.nav_voiceEnrollment)
        }

        // Quick Actions: Verify (Deepfake Voice Detection & Verification)
        val btnVerify = view.findViewById<LinearLayout>(R.id.btn_verify)
        btnVerify.setOnClickListener {
            findNavController().navigate(R.id.nav_voiceVerification)
        }

        // Quick Actions: Simulate Fake Call
        val btnSimulateCall = view.findViewById<LinearLayout>(R.id.btn_simulate_call)
        btnSimulateCall?.setOnClickListener {
            val intent = Intent(requireContext(), MockCallActivity::class.java)
            startActivity(intent)
        }

        // Quick Actions: History → Calls tab
        val btnHistory = view.findViewById<LinearLayout>(R.id.btn_history)
        btnHistory?.setOnClickListener {
            findNavController().navigate(R.id.nav_calls)
        }

        adapter = RecentActivityAdapter()
        rvActivity.adapter = adapter

        fetchDashboardData()
    }

    private fun fetchDashboardData() {
        pbLoading.visibility = View.VISIBLE
        rvActivity.visibility = View.GONE

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val response = RetrofitClient.instance.getDashboardData()
                withContext(Dispatchers.Main) {
                    pbLoading.visibility = View.GONE
                    if (response.isSuccessful && response.body() != null) {
                        val data = response.body()!!
                        adapter.submitList(data.recent_activity)
                        rvActivity.visibility = View.VISIBLE
                    } else {
                        // Show mock data when backend is not running
                        showMockData()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    pbLoading.visibility = View.GONE
                    // Show mock data for offline functionality
                    showMockData()
                }
            }
        }
    }

    private fun showMockData() {
        rvActivity.visibility = View.VISIBLE
        val mockItems = listOf(
            com.voice.shield.api.ActivityItem("1", "Unknown Caller", "+91 98765 43210", "Blocked", "2 min ago"),
            com.voice.shield.api.ActivityItem("2", "Bhavika Bhoir", "+91 87654 32100", "Safe", "15 min ago"),
            com.voice.shield.api.ActivityItem("3", "Spam Risk", "+91 76543 21000", "Blocked", "1 hour ago"),
            com.voice.shield.api.ActivityItem("4", "Mom", "+91 99999 88888", "Verified", "3 hours ago")
        )
        adapter.submitList(mockItems)
    }
}
