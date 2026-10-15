"""The model chooses semantic arguments; fixed adapters own hardware actions."""

SKILLS = [
    {"name": "go_to_region", "arguments": ["region_query"],
     "description": "到语义区域附近；雷达避障，定位波动时沿用估计并换视角恢复"},
    {"name": "find_and_approach", "arguments": ["object_query", "search_near_query"],
     "description": "区域内逐站多角度搜索，单摄发现后对准双摄示教工作区"},
    {"name": "model_grasp", "arguments": ["object_query"],
     "description": "模型重新确认双摄物品中心；闭爪后抬臂，不松爪"},
    {"name": "carry_to_region", "arguments": ["destination_query", "selection"],
     "description": "抓取后按当前位置的可达路径选择最近地点；携带期间禁止HOME和重复抓取"},
    {"name": "place_on_floor", "arguments": ["destination_query"],
     "description": "地点旁或原地地面放置；原地放下无需地点、定位或导航。复用本次抓取落地姿态，下臂后松爪再抬臂；不支持桌面/椅面高度"},
]


def is_in_place_release(instruction):
    """Recognize a complete release-only request, not navigation or a question."""
    import re
    text = re.sub(r"\s+", "", str(instruction or "")).strip("。！!，,").lower()
    if text in {"putitdown", "putitdownhere", "placeithere"}:
        return True
    prefix = r"(?:请|麻烦|帮我|现在|直接|就|先|再|给我)*"
    subject = (r"(?:(?:把|将)?(?:(?:手里|夹爪里|夹子中|夹爪中|夹着|已夹住|已经夹住)(?:的)?)?"
               r"(?:这个|那个|这块|那块|一块|一个)?(?:红色|绿色|黄色|蓝色)?"
               r"(?:目标物品|小物块|小木块|小方块|物块|木块|方块|积木|物品|物体|东西))?")
    location = r"(?:原地|就地|在这里|在这儿|在当前位置)?"
    release = (r"(?:放下来|放下|放到地面上?|放在地面上?|放到地上|放在地上|放在这里|放到这里|放在这儿|"
               r"放在当前位置|放到当前位置|放在原地|放回地面)")
    return bool(re.fullmatch(prefix + subject + prefix + location + release + subject
                             + r"(?:到地上|在地上)?(?:一下|吧)?", text))


def placement_selection(instruction, contract):
    import re
    query = str(contract.get("destination_query") or "").strip()
    # Only the destination clause can request nearest, not the pickup anchor.
    release_clause = re.split(r"然后|之后|接着|随后|[，,；;]", instruction)[-1]
    if is_in_place_release(release_clause):
        return {"query": "原地", "landmark_id": None, "selection": "in_place",
                "reference": "current_pose", "surface": "floor", "allow_chassis_motion": False,
                "integration_workflow": "robot_skills_v1"}
    destination_clause = re.split(r"放到|放在|搬到|送到|放置在", instruction)[-1]
    nearest = any(word in query or word in destination_clause for word in ("最近", "nearest", "closest"))
    surface = str(contract.get("placement_surface") or "floor")
    if re.search(r"(?:椅子|椅面|桌子|桌面|柜子|柜面|台面)(?:的)?(?:上|上面|顶部)", destination_clause):
        surface = "unsupported_height"
    return {"query": query, "landmark_id": None if nearest else contract.get("destination_landmark_id") or None,
            "selection": "nearest_reachable" if nearest else "named_region",
            "reference": "current_pose_after_pickup", "surface": surface,
            "integration_workflow": "robot_skills_v1"}
