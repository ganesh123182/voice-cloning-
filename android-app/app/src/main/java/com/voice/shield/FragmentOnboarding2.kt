package com.voice.shield

import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController

class FragmentOnboarding2 : Fragment(R.layout.fragment_onboarding2) {

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val btnNext = view.findViewById<ImageButton>(R.id.btn_next)
        val btnSkip = view.findViewById<TextView>(R.id.btn_skip)

        btnNext.setOnClickListener {
            findNavController().navigate(R.id.action_onboarding2_to_login)
        }

        btnSkip.setOnClickListener {
            findNavController().navigate(R.id.action_onboarding2_to_login)
        }
    }
}
