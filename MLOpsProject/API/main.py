import json
from dataclasses import dataclass, field
import random
from fastapi import FastAPI, HTTPException, Response

app = FastAPI()


@dataclass
class Channel:
    id: str
    name: str
    tags: list[str] = field(default_factory=list)
    description: str = ""


channels: dict[str, Channel] = {}

with open("channels.json", encoding="utf8") as file:
    channels_raw = json.load(file)
    for channel_raw in channels_raw:
        channel = Channel(**channel_raw)
        channels[channel.id] = channel


@app.get("/")
def read_root() -> Response:
    return Response("The server is running.")


@app.get("/channels/{channel_id}", response_model=Channel)
def read_item(channel_id: str) -> Channel:
    if channel_id not in channels:
        raise HTTPException(status_code=404, detail="Channel not found")
    else:
        rand_val = random.randint(0, 2)
        if rand_val == 0:
            win_id = "ct_win"
        else:
            win_id = "t_win"
    return channels[win_id]
