import json
from dataclasses import dataclass, field
import random
from fastapi import FastAPI, HTTPException, Response
from data.raw_2_clean import Raw2Clean
from predict_model import predict

app = FastAPI()
r2c = Raw2Clean()
model_path = "models/trained_model.pt"

@dataclass
class Channel:
    id: str
    name: str
    tags: list[str] = field(default_factory=list)
    description: str = ""


channels: dict[str, Channel] = {}

with open("./API/channels.json", encoding="utf8") as file:
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
        random_value = random.randint(0, 1)

        if random_value == 1:
            data_raw = [94.96, 7.0, 4.0, 'de_dust2', False, 
                        500.0, 500.0, 466.0, 300.0, 18950.0, 
                        10600.0, 4.0, 2.0, 4.0, 5.0, 
                        5.0, 2.0, 0.0, 1.0, 0.0, 
                        2.0, 0.0, 0.0, 0.0, 0.0, 
                        1.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 2.0, 2.0, 0.0, 
                        0.0, 2.0, 0.0, 1.0, 2.0, 
                        0.0, 0.0, 0.0, 0.0, 2.0, 
                        0.0, 5.0, 2.0, 3.0, 0.0, 
                        3.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 'CT']
        else:
            data_raw = [114.97, 2.0, 0.0, 'de_dust2', False, 
                500.0, 500.0, 496.0, 500.0, 2200.0, 
                1000.0, 4.0, 5.0, 2.0, 5.0, 
                5.0, 0.0, 5.0, 0.0, 0.0, 
                1.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 1.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                5.0, 0.0, 0.0, 0.0, 0.0, 
                2.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 1.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 4.0, 0.0, 0.0, 0.0, 
                1.0, 0.0, 0.0, 0.0, 1.0, 
                0.0, 4.0, 5.0, 4.0, 5.0, 
                2.0, 0.0, 0.0, 1.0, 0.0, 
                1.0, 'T']

        dataloader = r2c.clean_data(data_raw)

        predicted = predict(model_path,dataloader)
        print(predicted)
        if predicted[0].argmax().item() == 0:
            win_id = "ct_win"
        else:
            win_id = "t_win"

    return channels[win_id]

