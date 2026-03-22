class VideoMemory:
    def __init__(self, max_rolling_size: int = 10):
        # Built once in the pre-pass before the main loop
        self.global_registry = {"CHARACTERS": {}, "LOCATIONS": {}, "PROPS": {}}
        # Rolling window of recent shot summaries
        self.rolling_shot_cache = []
        self.max_rolling_size = max_rolling_size

    def add_to_shot_cache(self, shot_id: int, shot_summary: str) -> None:
        entry = f"Shot {shot_id}: {shot_summary}"
        self.rolling_shot_cache.append(entry)
        if len(self.rolling_shot_cache) > self.max_rolling_size:
            self.rolling_shot_cache.pop(0)

    def update_registry(self, new_entities: dict) -> None:
        for category, entities in new_entities.items():
            if category in self.global_registry and isinstance(entities, dict):
                self.global_registry[category].update(entities)

    def format_registry(self) -> str:
        if not any(self.global_registry.values()):
            return "No persistent entities identified yet."
        output = "PERSISTENT STORY LEGEND (Do not re-describe these):\n"
        for category, items in self.global_registry.items():
            if items:
                output += f"--- {category} ---\n"
                for eid, details in items.items():
                    output += f"{eid}: {details}\n"
        return output.rstrip()

    def format_rolling_cache(self) -> str:
        if not self.rolling_shot_cache:
            return "Beginning of sequence."
        return "\n".join(self.rolling_shot_cache)
