package com.voice.shield

import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment

class FragmentCalls : Fragment(R.layout.fragment_calls) {

    // Full list of call entries (category is index 3)
    private val allCalls = listOf(
        arrayOf("Unknown Caller — Blocked", "+91 98765 43210", "2 min ago", "Blocked"),
        arrayOf("Bhavika Bhoir — Safe", "+91 87654 32100", "15 min ago", "Safe"),
        arrayOf("Spam Risk — Blocked", "+91 76543 21000", "1 hour ago", "Blocked"),
        arrayOf("Mom — Verified", "+91 99999 88888", "3 hours ago", "Safe"),
        arrayOf("Bank Support — Safe", "+91 18001 80000", "Yesterday", "Safe"),
        arrayOf("AI Scam Caller", "+91 70001 23456", "Yesterday", "Suspicious"),
        arrayOf("Deepfake Voice — Flagged", "+91 65432 10000", "2 days ago", "Suspicious")
    )

    private var listContainer: android.view.ViewGroup? = null

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Find the placeholder text view and its parent container
        val placeholder = findTextViewByText(view, "Call history items will appear here...")
        placeholder?.let {
            listContainer = it.parent as? android.view.ViewGroup
            listContainer?.removeView(it)
        }

        // Populate with all calls initially
        populateCallList(allCalls)

        // Wire up filter pills
        val filterContainer = findFilterContainer(view)
        if (filterContainer != null) {
            for (i in 0 until filterContainer.childCount) {
                val chip = filterContainer.getChildAt(i) as? TextView ?: continue
                chip.setOnClickListener { clickedChip ->
                    // Reset all chips to inactive style
                    for (j in 0 until filterContainer.childCount) {
                        val c = filterContainer.getChildAt(j) as? TextView ?: continue
                        c.setBackgroundColor(ContextCompat.getColor(requireContext(), android.R.color.transparent))
                        c.setTextColor(ContextCompat.getColor(requireContext(), R.color.text_secondary))
                    }
                    // Highlight selected chip
                    val selected = clickedChip as TextView
                    selected.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.trustvoice_primary))
                    selected.setTextColor(ContextCompat.getColor(requireContext(), R.color.text_white))

                    // Filter
                    val filterText = selected.text.toString()
                    val filtered = when (filterText) {
                        "Safe" -> allCalls.filter { it[3] == "Safe" }
                        "Warning" -> allCalls.filter { it[3] == "Blocked" }
                        "Suspicious" -> allCalls.filter { it[3] == "Suspicious" }
                        else -> allCalls // "All"
                    }
                    populateCallList(filtered)
                }
            }
        }
    }

    private fun populateCallList(calls: List<Array<String>>) {
        val parent = listContainer ?: return
        parent.removeAllViews()
        for (call in calls) {
            val item = layoutInflater.inflate(R.layout.item_recent_activity, parent, false)
            item.findViewById<TextView>(R.id.tv_title)?.text = call[0]
            item.findViewById<TextView>(R.id.tv_subtitle)?.text = call[1]
            item.findViewById<TextView>(R.id.tv_timestamp)?.text = call[2]
            val badge = item.findViewById<TextView>(R.id.tv_badge)
            badge?.text = call[3]
            // Color the badge based on category
            when (call[3]) {
                "Safe" -> {
                    badge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_safe))
                    badge?.setBackgroundResource(R.drawable.bg_badge_safe)
                }
                "Blocked" -> {
                    badge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_suspicious))
                    badge?.setBackgroundResource(R.drawable.bg_badge_suspicious)
                }
                "Suspicious" -> {
                    badge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_suspicious))
                    badge?.setBackgroundResource(R.drawable.bg_badge_suspicious)
                }
            }
            parent.addView(item)
        }
        // Update footer stat
        view?.let { root ->
            findTextViewByText(root, "Recent checks: 5")?.text = "Recent checks: ${calls.size}"
        }
    }

    /** Find the LinearLayout inside the HorizontalScrollView that holds the filter chips */
    private fun findFilterContainer(root: View): LinearLayout? {
        if (root is android.widget.HorizontalScrollView) {
            val child = (root as android.view.ViewGroup).getChildAt(0)
            if (child is LinearLayout) return child
        }
        if (root is android.view.ViewGroup) {
            for (i in 0 until root.childCount) {
                val found = findFilterContainer(root.getChildAt(i))
                if (found != null) return found
            }
        }
        return null
    }

    private fun findTextViewByText(root: View, text: String): TextView? {
        if (root is TextView && root.text.toString() == text) return root
        if (root is android.view.ViewGroup) {
            for (i in 0 until root.childCount) {
                val found = findTextViewByText(root.getChildAt(i), text)
                if (found != null) return found
            }
        }
        return null
    }
}
