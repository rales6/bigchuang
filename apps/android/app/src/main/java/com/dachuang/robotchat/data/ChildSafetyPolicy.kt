package com.dachuang.robotchat.data

data class SafetyResult(val allowed: Boolean, val reason: String = "")

object ChildSafetyPolicy {
    private val restrictedWords = listOf(
        "高速", "加速", "全速", "前进", "后退", "倒车", "旋转", "转弯",
        "抓取", "夹取", "抬起", "搬运", "移动到", "机械臂",
        "fast", "forward", "backward", "rotate", "grab", "arm",
    )
    private val alwaysSafeStopWords = listOf("停止", "停下", "急停", "stop")

    fun validate(text: String): SafetyResult {
        val normalized = text.trim().lowercase()
        if (normalized.isEmpty()) return SafetyResult(false, "内容不能为空")
        if (normalized.length > 80) return SafetyResult(false, "少儿模式每条消息最多 80 个字")
        if (alwaysSafeStopWords.any(normalized::contains)) return SafetyResult(true)
        val matched = restrictedWords.firstOrNull(normalized::contains)
        return if (matched == null) {
            SafetyResult(true)
        } else {
            SafetyResult(false, "少儿模式不能发送包含“$matched”的运动控制指令，请由监护人解锁")
        }
    }
}
