package com.dachuang.robotchat.network

import com.dachuang.robotchat.model.RobotSnapshot
import com.dachuang.robotchat.model.RobotServiceState
import com.dachuang.robotchat.model.RobotTaskPlan
import com.dachuang.robotchat.model.RobotTaskState
import com.dachuang.robotchat.model.TaskPlanStep
import org.json.JSONArray
import org.json.JSONObject

object RobotStateParser {
    fun parse(json: String): RobotSnapshot {
        val root = JSONObject(json)
        val camera = root.optJSONObject("camera") ?: JSONObject()
        val task = root.optJSONObject("robot_task") ?: JSONObject()
        val command = root.optJSONObject("robot_command") ?: JSONObject()
        val queue = root.optJSONArray("target_queue") ?: JSONArray()

        var done = 0
        var activeTarget = ""
        for (index in 0 until queue.length()) {
            val item = queue.optJSONObject(index) ?: continue
            val status = item.optString("status")
            if (status == "done" || status == "completed" || item.optBoolean("done")) done += 1
            if (status == "active" || item.optBoolean("active")) {
                activeTarget = firstText(item, "target_name", "name", "instruction")
            }
        }
        val total = queue.length().takeIf { it > 0 } ?: task.optInt("queue_total", 0)
        val completed = task.optBoolean("completed", false)
        val phase = task.optString("phase", "idle")
        val progress = when {
            completed -> 1f
            total > 0 -> (done.toFloat() / total.toFloat()).coerceIn(0f, 1f)
            else -> phaseProgress(phase)
        }

        val reply = firstText(root, "reply", "assistant_message", "message", "text_response")
        val replyId = firstText(root, "reply_id", "message_id").ifEmpty {
            if (reply.isEmpty()) "" else "${root.optLong("frame_index", 0)}:${reply.hashCode()}"
        }
        val obstacles = root.optJSONObject("obstacles")
        val lidar = root.optJSONObject("lidar")
        val taskPlan = root.optJSONObject("task_plan") ?: JSONObject()
        val services = root.optJSONObject("services")?.optJSONObject("process") ?: JSONObject()
        val planStepsJson = taskPlan.optJSONArray("steps") ?: JSONArray()
        val planSteps = buildList {
            for (index in 0 until planStepsJson.length()) {
                val step = planStepsJson.optJSONObject(index) ?: continue
                add(
                    TaskPlanStep(
                        id = step.optString("id", "step_${index + 1}"),
                        title = step.optString("title", step.optString("action", "任务步骤")),
                        detail = step.optString("detail", ""),
                        status = step.optString("status", "pending"),
                    ),
                )
            }
        }

        return RobotSnapshot(
            connected = true,
            cameraConnected = camera.optBoolean("connected", false),
            groundingStatus = root.optString("grounding_status", "已连接"),
            emergencyStop = root.optBoolean("emergency_stop", false),
            obstacleStatus = obstacles?.optString("status", "未运行") ?: "未运行",
            lidarStatus = lidar?.optString("status", "未运行") ?: "未运行",
            robotCommand = firstText(command, "action", "subsystem").ifEmpty { "idle" },
            commandReason = command.optString("reason", ""),
            task = RobotTaskState(
                active = task.optBoolean("active", false),
                taskType = task.optString("task_type", "none"),
                phase = phase,
                phaseLabel = firstText(task, "phase_label", "status").ifEmpty { localizePhase(phase) },
                completed = completed,
                progress = progress,
                completedTargets = done,
                totalTargets = total,
                currentTarget = activeTarget,
            ),
            taskPlan = RobotTaskPlan(
                instruction = taskPlan.optString("instruction", ""),
                status = taskPlan.optString("status", "idle"),
                summary = taskPlan.optString("summary", ""),
                completedSteps = taskPlan.optInt("completed_steps", 0),
                totalSteps = taskPlan.optInt("total_steps", planSteps.size),
                mapUrl = taskPlan.optString("map_url", "/task/map.png"),
                updatedAt = taskPlan.optLong("updated_at", 0L),
                steps = planSteps,
            ),
            service = RobotServiceState(
                kind = services.optString("kind", "idle"),
                running = services.optBoolean("running", false),
                logPath = services.optString("log", ""),
                lastExitCode = if (services.has("last_exit_code") && !services.isNull("last_exit_code")) {
                    services.optInt("last_exit_code")
                } else null,
            ),
            replyText = reply,
            replyId = replyId,
            rawJson = json,
        )
    }

    private fun firstText(objectValue: JSONObject, vararg keys: String): String {
        for (key in keys) {
            val value = objectValue.opt(key)
            when (value) {
                is String -> if (value.isNotBlank()) return value
                is JSONObject -> {
                    val nested = firstText(value, "text", "content", "message")
                    if (nested.isNotBlank()) return nested
                }
            }
        }
        return ""
    }

    private fun phaseProgress(phase: String): Float = when (phase.lowercase()) {
        "idle" -> 0f
        "grounding", "searching", "locating" -> .15f
        "approaching", "navigating", "moving" -> .35f
        "aligning" -> .55f
        "grasping", "picking" -> .7f
        "disposing", "placing", "returning" -> .85f
        "done", "completed" -> 1f
        else -> .1f
    }

    private fun localizePhase(phase: String): String = when (phase.lowercase()) {
        "idle" -> "暂无任务"
        "grounding", "searching", "locating" -> "正在寻找目标"
        "approaching", "navigating", "moving" -> "正在移动"
        "aligning" -> "正在对准目标"
        "grasping", "picking" -> "正在抓取"
        "disposing", "placing" -> "正在放置"
        "returning" -> "正在返回"
        "done", "completed" -> "任务已完成"
        else -> phase.ifBlank { "状态未知" }
    }
}
