package com.voice.shield

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.voice.shield.api.ActivityItem

class RecentActivityAdapter : RecyclerView.Adapter<RecentActivityAdapter.ViewHolder>() {

    private val items = mutableListOf<ActivityItem>()

    fun submitList(newItems: List<ActivityItem>) {
        items.clear()
        items.addAll(newItems)
        notifyDataSetChanged()
    }

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val ivIcon: ImageView = view.findViewById(R.id.iv_status_icon)
        val tvTitle: TextView = view.findViewById(R.id.tv_title)
        val tvSubtitle: TextView = view.findViewById(R.id.tv_subtitle)
        val tvTimestamp: TextView = view.findViewById(R.id.tv_timestamp)
        val tvBadge: TextView = view.findViewById(R.id.tv_badge)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_recent_activity, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        val context = holder.itemView.context

        holder.tvSubtitle.text = item.phone_number
        holder.tvTimestamp.text = item.timestamp
        holder.tvBadge.text = item.status

        when (item.status.lowercase()) {
            "safe" -> {
                holder.tvTitle.text = "Call analyzed — Safe"
                holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_safe))
                holder.tvBadge.setBackgroundResource(R.drawable.bg_badge_safe)
            }
            "blocked", "suspicious" -> {
                holder.tvTitle.text = "Suspicious voice detected"
                holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_suspicious))
                holder.tvBadge.setBackgroundResource(R.drawable.bg_badge_suspicious)
                holder.ivIcon.setColorFilter(ContextCompat.getColor(context, R.color.status_suspicious))
            }
            "verified" -> {
                holder.tvTitle.text = "Verification completed"
                holder.tvBadge.setTextColor(ContextCompat.getColor(context, R.color.status_safe))
                holder.tvBadge.setBackgroundResource(R.drawable.bg_badge_safe)
            }
            else -> {
                holder.tvTitle.text = item.status
            }
        }
    }

    override fun getItemCount() = items.size
}
