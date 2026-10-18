"""Live app: predict a household's choice on one visit, and inspect logged runs.

Run from the project folder:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs to change inputs and see new results.

The app only predicts with models that already exist; it never trains. It reads
runs.db in read-only mode, so it cannot add or change runs.
"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

import baseline
from experiments import m0, m1, m2, m3


ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "runs.db"
MODEL_DIR = ROOT / "models"


@dataclass
class Model:
    predict: Callable
    run_id: Optional[int]
    provisional: bool
    description: str


# Each fitted model is added here once its run is logged and its file in models/ is committed.
# Provisional means it has not been evaluated on the check set yet.
MODELS = {
    "baseline": Model(
        predict=baseline.predict,
        run_id=1,
        provisional=False,
        description="Starter baseline from baseline.py. Its weights are hand-set, not learned from data, "
        "and its reason is the last rule that matched, not the largest effect.",
    ),
}
if (MODEL_DIR / "M0.json").exists():
    saved = json.loads((MODEL_DIR / "M0.json").read_text(encoding="utf-8"))
    MODELS["M0"] = Model(
        predict=m0.predictor(saved["shares"]),
        run_id=saved["run_id"],
        provisional=True,
        description="M0: every visit gets the choice shares of the 200 build households. "
        "It ignores all inputs, so changing the input does not change its answer.",
    )
if (MODEL_DIR / "M1.json").exists():
    saved = json.loads((MODEL_DIR / "M1.json").read_text(encoding="utf-8"))
    MODELS["M1"] = Model(
        predict=m1.predictor(saved),
        run_id=saved["run_id"],
        provisional=True,
        description="M1: household traits only (usual brand, budget, income band, cleaning spend, bathrooms); "
        "ignores the offer and history. Needs budget, income_band and bathrooms (1 to 3); a missing spend is "
        "filled like a blank. Its run was inconclusive: it did not clearly beat M0 and lost to the baseline.",
    )
if (MODEL_DIR / "M2.json").exists():
    saved = json.loads((MODEL_DIR / "M2.json").read_text(encoding="utf-8"))
    MODELS["M2"] = Model(
        predict=m2.predictor(saved),
        run_id=saved["run_id"],
        provisional=True,
        description="M2: the levers (all four prices, entrant message, channel) plus usual brand and spend -> "
        "entrant; ignores history. Needs cleaning_spend_usd, or bathrooms (1 to 3) to fill a blank spend. "
        "It beat M1r in 4 of 5 folds and the baseline on average, but only in 3 of 5 folds.",
    )
if (MODEL_DIR / "M3.json").exists():
    saved = json.loads((MODEL_DIR / "M3.json").read_text(encoding="utf-8"))
    MODELS["M3"] = Model(
        predict=m3.predictor(saved),
        run_id=saved["run_id"],
        provisional=True,
        description="M3: M2 plus purchase history from the visits in 'history' (times each product was bought, "
        "the most recent product bought, whether the previous visit was a purchase, first visit). Same household "
        "inputs as M2. Best model so far: it beat M2 in 4 of 5 folds (a small gain) and the baseline in 4 of 5 folds.",
    )
DEFAULT_MODEL = "baseline"
ModelName = Enum("ModelName", {name: name for name in MODELS}, type=str)


class Household(BaseModel):
    budget: Optional[Literal["tight", "middle", "flexible"]] = None
    usual_brand: Literal["standard", "budget", "scented", ""] = ""
    income_band: Optional[Literal["under_50k", "50k_to_100k", "over_100k"]] = None
    cleaning_spend_usd: Optional[float] = Field(None, ge=0)
    bathrooms: Optional[int] = Field(None, ge=1)
    note: str = ""


class Offer(BaseModel):
    entrant_price: float = Field(gt=0)
    entrant_message: Literal["cleaning", "low_odor", "environment", ""] = ""
    standard_price: float = Field(gt=0)
    budget_price: float = Field(gt=0)
    scented_price: float = Field(gt=0)


class PastVisit(BaseModel):
    choice: Literal["entrant", "standard", "budget", "scented", "none"]
    channel: Optional[Literal["online", "retail"]] = None
    said_why: str = ""
    after_use_note: str = ""


class PredictRequest(BaseModel):
    household: Household
    offer: Offer
    channel: Literal["online", "retail"]
    history: list[PastVisit] = Field(default_factory=list, description="Earlier visits, oldest first.")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "household": {"budget": "middle", "usual_brand": "standard", "income_band": "50k_to_100k",
                      "cleaning_spend_usd": 13, "bathrooms": 1, "note": ""},
        "offer": {"entrant_price": 4.49, "entrant_message": "environment", "standard_price": 3.99,
                  "budget_price": 2.99, "scented_price": 4.29},
        "channel": "online",
        "history": [{"choice": "standard", "channel": "retail"}],
    }]})


class Prediction(BaseModel):
    model: str
    run_id: Optional[int]
    provisional: bool
    probabilities: dict[str, float]
    most_likely: str
    reason: str


app = FastAPI(
    title="Cleaner choice challenge",
    description="Predict which cleaner a household chooses on one visit, and inspect logged experiment runs.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/models")
def list_models():
    return [
        {"name": name, "default": name == DEFAULT_MODEL, "run_id": model.run_id,
         "provisional": model.provisional, "description": model.description}
        for name, model in MODELS.items()
    ]


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest, model: ModelName = ModelName(DEFAULT_MODEL)):
    chosen = MODELS[model.value]
    offer = request.offer
    offers = [
        {"product_id": "entrant", "price": offer.entrant_price, "message": offer.entrant_message},
        {"product_id": "standard", "price": offer.standard_price},
        {"product_id": "budget", "price": offer.budget_price},
        {"product_id": "scented", "price": offer.scented_price},
    ]
    visit = {"channel": request.channel, "visit_number": len(request.history) + 1}
    history = [past.model_dump() for past in request.history]
    try:
        result = chosen.predict(request.household.model_dump(), offers, visit, history)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"{model.value} cannot use this input: {error}")
    probabilities = result["probabilities"]
    return Prediction(
        model=model.value,
        run_id=chosen.run_id,
        provisional=chosen.provisional,
        probabilities=probabilities,
        most_likely=max(probabilities, key=probabilities.get),
        reason=result["reason"],
    )


def read_only_database():
    if not DATABASE.exists():
        raise HTTPException(status_code=503, detail="runs.db is not available.")
    connection = sqlite3.connect(f"{DATABASE.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


@app.get("/runs")
def list_runs():
    with closing(read_only_database()) as connection:
        rows = connection.execute("SELECT id, name, status, started_at, backfilled FROM runs ORDER BY id").fetchall()
    return [dict(row) for row in rows]


@app.get("/runs/{run_id}")
def show_run(run_id: int):
    with closing(read_only_database()) as connection:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No run with ID {run_id}.")
    run = dict(row)
    for key in ("inputs", "data_version"):
        if run.get(key):
            run[key] = json.loads(run[key])
    return run
