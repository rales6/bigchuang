package com.dachuang.robotchat

import com.dachuang.robotchat.network.RobotStateParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RobotStateParserTest {
    @Test
    fun parsesCurrentServerStateAndQueueProgress() {
        val state = RobotStateParser.parse(
            """
            {
              "frame_index": 42,
              "grounding_status": "tracking bottle",
              "camera": {"connected": true, "fps": 9.8},
              "obstacles": {"status": "clear"},
              "lidar": {"status": "ready"},
              "robot_task": {
                "active": true,
                "task_type": "pick_multiple",
                "phase": "approaching",
                "phase_label": "Approaching target",
                "completed": false
              },
              "robot_command": {"action": "forward", "reason": "target ahead"},
              "target_queue": [
                {"target_name": "first bottle", "status": "done"},
                {"target_name": "second bottle", "status": "active"}
              ],
              "assistant_message": "正在处理第二个水瓶",
              "reply_id": "reply-7"
            }
            """.trimIndent(),
        )
        assertTrue(state.connected)
        assertTrue(state.cameraConnected)
        assertEquals("forward", state.robotCommand)
        assertEquals(1, state.task.completedTargets)
        assertEquals(2, state.task.totalTargets)
        assertEquals(.5f, state.task.progress)
        assertEquals("second bottle", state.task.currentTarget)
        assertEquals("正在处理第二个水瓶", state.replyText)
    }

    @Test
    fun acceptsNestedReplyObject() {
        val state = RobotStateParser.parse("""{"reply":{"text":"任务完成"}}""")
        assertEquals("任务完成", state.replyText)
    }

    @Test
    fun parsesVisibleTaskPlanAndManagedService() {
        val state = RobotStateParser.parse(
            """
            {
              "task_plan": {
                "instruction": "去柜子边找红色物块",
                "status": "running",
                "summary": "正在前往第二个观察点",
                "completed_steps": 2,
                "total_steps": 4,
                "map_url": "/task/map.png",
                "updated_at": 123456,
                "steps": [
                  {"id":"s1","title":"全局定位","detail":"定位已接受","status":"completed"},
                  {"id":"s2","title":"前往柜子附近","detail":"传统算法执行航点","status":"active"}
                ]
              },
              "services": {
                "process": {"kind":"object_search","running":true,"log":"service_runs/run.log","last_exit_code":null}
              }
            }
            """.trimIndent(),
        )
        assertEquals("去柜子边找红色物块", state.taskPlan.instruction)
        assertEquals("active", state.taskPlan.steps[1].status)
        assertEquals(123456L, state.taskPlan.updatedAt)
        assertEquals("object_search", state.service.kind)
        assertTrue(state.service.running)
    }
}
