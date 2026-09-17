package com.voice.shield

import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController

class FragmentWelcome : Fragment(R.layout.fragment_welcome) {

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val btnGetStarted = view.findViewById<Button>(R.id.btn_get_started)
        btnGetStarted.setOnClickListener {
            findNavController().navigate(R.id.action_welcome_to_onboarding1)
        }
    }
}
