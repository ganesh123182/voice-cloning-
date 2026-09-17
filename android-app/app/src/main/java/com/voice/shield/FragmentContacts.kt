package com.voice.shield

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.fragment.app.Fragment

class FragmentContacts : Fragment(R.layout.fragment_contacts) {
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val btnAddContact = view.findViewById<Button>(R.id.btn_add_contact)
        btnAddContact?.setOnClickListener {
            Toast.makeText(requireContext(), "Add Trusted Contact feature coming soon!", Toast.LENGTH_SHORT).show()
        }
    }
}
