import json
from typing import Any, Dict

from mcap.writer import Writer
from mcap.records import Message

NS_PER_SEC = 1_000_000_000

def sec_to_ns(seconds: float) -> int:
    return int(seconds * NS_PER_SEC)

class McapLogger:
    """
    MCAP logger compatible with older mcap Writer API
    (register_schema(name, encoding, data)).
    """
    def __init__(self, file_path: str):
        self._f = open(file_path, "wb")
        self.writer = Writer(self._f)
        self.writer.start()

        self.schemas: Dict[str, int] = {}
        self.channels: Dict[str, int] = {}
        self._seq: Dict[str, int] = {}

    def register_schema(self, schema_name: str, schema_def: Dict[str, Any]) -> int:
        schema_id = self.writer.register_schema(
            schema_name,
            "jsonschema",
            json.dumps(schema_def).encode("utf-8"),
        )
        self.schemas[schema_name] = schema_id
        return schema_id

    def register_channel(self, topic: str, schema_id: int) -> int:
        channel_id = self.writer.register_channel(
            topic,
            "json",
            schema_id,
            {}
        )
        self.channels[topic] = channel_id
        self._seq[topic] = 0
        return channel_id

    def write(self, topic: str, t_ns: int, msg: Dict[str, Any]) -> None:
        if topic not in self.channels:
            raise ValueError(f"Channel '{topic}' not registered")
    
        seq = self._seq[topic]
        self._seq[topic] += 1
    
        data = json.dumps(msg).encode("utf-8")
    
        # Old Writer API: add_message(channel_id, log_time, data, publish_time, sequence)
        self.writer.add_message(
            self.channels[topic],  # channel_id
            t_ns,                  # log_time
            data,                  # data (bytes)
            t_ns,                  # publish_time
            seq,                   # sequence
        )


    def close(self) -> None:
        self.writer.finish()
        self._f.close()
