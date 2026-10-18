import hashlib
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None

from baseline import predict


REQUEST = {
    "household": {"budget": "middle", "usual_brand": "standard"},
    "offer": {"entrant_price": 4.49, "entrant_message": "cleaning", "standard_price": 3.99,
              "budget_price": 2.99, "scented_price": 4.29},
    "channel": "retail",
}


@unittest.skipIf(TestClient is None, "App dependencies are not installed (pip install -r requirements-dev.txt).")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.main import DATABASE, app
        cls.client = TestClient(app)
        cls.database = DATABASE

    def test_home_redirects_to_docs(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")

    def test_models_lists_baseline_as_default(self):
        models = self.client.get("/models").json()
        self.assertIn({"name": "baseline", "default": True}, [{k: m[k] for k in ("name", "default")} for m in models])

    def test_prediction_covers_every_option(self):
        result = self.client.post("/predict", json=REQUEST).json()
        self.assertEqual(set(result["probabilities"]), {"entrant", "standard", "budget", "scented", "none"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertEqual(result["model"], "baseline")
        self.assertEqual(result["run_id"], 1)

    def test_prediction_matches_baseline_called_directly(self):
        result = self.client.post("/predict", json=REQUEST).json()
        offers = [
            {"product_id": "entrant", "price": 4.49, "message": "cleaning"},
            {"product_id": "standard", "price": 3.99},
            {"product_id": "budget", "price": 2.99},
            {"product_id": "scented", "price": 4.29},
        ]
        household = {"budget": "middle", "usual_brand": "standard"}
        direct = predict(household, offers, {"channel": "retail"}, [])
        for option, probability in direct["probabilities"].items():
            self.assertAlmostEqual(result["probabilities"][option], probability)
        self.assertEqual(result["reason"], direct["reason"])

    def test_changed_input_changes_result(self):
        first = self.client.post("/predict", json=REQUEST).json()["probabilities"]["entrant"]
        cheaper = {**REQUEST, "offer": {**REQUEST["offer"], "entrant_price": 3.49}}
        second = self.client.post("/predict", json=cheaper).json()["probabilities"]["entrant"]
        self.assertGreater(second, first)

    def test_history_is_used(self):
        first = self.client.post("/predict", json=REQUEST).json()["probabilities"]["none"]
        after_purchase = {**REQUEST, "history": [{"choice": "standard"}]}
        second = self.client.post("/predict", json=after_purchase).json()["probabilities"]["none"]
        self.assertNotEqual(first, second)

    def test_invalid_inputs_are_rejected(self):
        bad_channel = {**REQUEST, "channel": "phone"}
        bad_price = {**REQUEST, "offer": {**REQUEST["offer"], "entrant_price": -1}}
        bad_choice = {**REQUEST, "history": [{"choice": "soap"}]}
        for body in (bad_channel, bad_price, bad_choice):
            self.assertEqual(self.client.post("/predict", json=body).status_code, 422)
        self.assertEqual(self.client.post("/predict?model=unknown", json=REQUEST).status_code, 422)

    def test_m0_returns_its_saved_shares_whatever_the_input(self):
        from app.main import MODELS, MODEL_DIR
        if "M0" not in MODELS:
            self.skipTest("models/M0.json is not present.")
        import json
        saved = json.loads((MODEL_DIR / "M0.json").read_text(encoding="utf-8"))
        first = self.client.post("/predict?model=M0", json=REQUEST).json()
        cheaper = {**REQUEST, "offer": {**REQUEST["offer"], "entrant_price": 3.49}, "history": [{"choice": "entrant"}]}
        second = self.client.post("/predict?model=M0", json=cheaper).json()
        self.assertEqual(first["probabilities"], saved["shares"])
        self.assertEqual(second["probabilities"], saved["shares"])
        self.assertEqual((first["model"], first["run_id"], first["provisional"]), ("M0", saved["run_id"], True))

    def test_m1_uses_household_traits_and_rejects_missing_ones(self):
        from app.main import MODELS, MODEL_DIR
        if "M1" not in MODELS:
            self.skipTest("models/M1.json is not present.")
        import json
        from experiments import m1
        saved = json.loads((MODEL_DIR / "M1.json").read_text(encoding="utf-8"))
        household = {"budget": "middle", "usual_brand": "standard", "income_band": "50k_to_100k",
                     "cleaning_spend_usd": 13, "bathrooms": 1}
        result = self.client.post("/predict?model=M1", json={**REQUEST, "household": household}).json()
        direct = m1.predictor(saved)({**household, "note": ""}, None, None, [])
        for option, probability in direct["probabilities"].items():
            self.assertAlmostEqual(result["probabilities"][option], probability)
        self.assertEqual((result["model"], result["run_id"], result["provisional"]), ("M1", saved["run_id"], True))
        richer = {**household, "cleaning_spend_usd": 40}
        changed = self.client.post("/predict?model=M1", json={**REQUEST, "household": richer}).json()
        self.assertNotEqual(changed["probabilities"], result["probabilities"])
        blank_spend = self.client.post("/predict?model=M1", json={**REQUEST, "household": {**household, "cleaning_spend_usd": None}})
        self.assertEqual(blank_spend.status_code, 200)
        for missing in ({**household, "budget": None}, {**household, "income_band": None}, {**household, "bathrooms": 4}):
            response = self.client.post("/predict?model=M1", json={**REQUEST, "household": missing})
            self.assertEqual(response.status_code, 422)
            self.assertIn("M1 cannot use this input", response.json()["detail"])

    def test_m2_uses_the_offer_and_rejects_unusable_input(self):
        from app.main import MODELS, MODEL_DIR
        if "M2" not in MODELS:
            self.skipTest("models/M2.json is not present.")
        import json
        from experiments import m2
        saved = json.loads((MODEL_DIR / "M2.json").read_text(encoding="utf-8"))
        household = {"budget": "middle", "usual_brand": "standard", "cleaning_spend_usd": 13, "bathrooms": 1}
        body = {**REQUEST, "household": household}
        result = self.client.post("/predict?model=M2", json=body).json()
        offers = [
            {"product_id": "entrant", "price": 4.49, "message": "cleaning"},
            {"product_id": "standard", "price": 3.99},
            {"product_id": "budget", "price": 2.99},
            {"product_id": "scented", "price": 4.29},
        ]
        direct = m2.predictor(saved)({**household, "income_band": None, "note": ""}, offers, {"channel": "retail"}, [])
        for option, probability in direct["probabilities"].items():
            self.assertAlmostEqual(result["probabilities"][option], probability)
        self.assertEqual((result["model"], result["run_id"], result["provisional"]), ("M2", saved["run_id"], True))
        cheaper = self.client.post("/predict?model=M2", json={**body, "offer": {**REQUEST["offer"], "entrant_price": 3.49}}).json()
        self.assertGreater(cheaper["probabilities"]["entrant"], result["probabilities"]["entrant"])
        blank_message = self.client.post("/predict?model=M2", json={**body, "offer": {**REQUEST["offer"], "entrant_message": ""}})
        self.assertEqual(blank_message.status_code, 200)
        no_spend_no_bathrooms = {**household, "cleaning_spend_usd": None, "bathrooms": None}
        response = self.client.post("/predict?model=M2", json={**REQUEST, "household": no_spend_no_bathrooms})
        self.assertEqual(response.status_code, 422)
        self.assertIn("M2 cannot use this input", response.json()["detail"])

    def test_m3_uses_history(self):
        from app.main import MODELS, MODEL_DIR
        if "M3" not in MODELS:
            self.skipTest("models/M3.json is not present.")
        import json
        from experiments import m3
        saved = json.loads((MODEL_DIR / "M3.json").read_text(encoding="utf-8"))
        household = {"budget": "middle", "usual_brand": "standard", "cleaning_spend_usd": 13, "bathrooms": 1}
        history = [{"choice": "standard"}, {"choice": "none"}, {"choice": "entrant"}]
        body = {**REQUEST, "household": household, "history": history}
        result = self.client.post("/predict?model=M3", json=body).json()
        offers = [
            {"product_id": "entrant", "price": 4.49, "message": "cleaning"},
            {"product_id": "standard", "price": 3.99},
            {"product_id": "budget", "price": 2.99},
            {"product_id": "scented", "price": 4.29},
        ]
        direct = m3.predictor(saved)({**household, "income_band": None, "note": ""}, offers, {"channel": "retail"}, history)
        for option, probability in direct["probabilities"].items():
            self.assertAlmostEqual(result["probabilities"][option], probability)
        self.assertEqual((result["model"], result["run_id"], result["provisional"]), ("M3", saved["run_id"], True))
        first_visit = self.client.post("/predict?model=M3", json={**body, "history": []}).json()
        self.assertNotEqual(first_visit["probabilities"], result["probabilities"])

    def test_runs_can_be_listed_and_inspected(self):
        runs = self.client.get("/runs").json()
        self.assertGreaterEqual(len(runs), 1)
        first = self.client.get(f"/runs/{runs[0]['id']}").json()
        self.assertEqual(first["id"], runs[0]["id"])
        self.assertIsInstance(first["inputs"], dict)
        self.assertEqual(self.client.get("/runs/999999").status_code, 404)

    def test_app_does_not_change_runs_database(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        self.client.get("/runs")
        self.client.get("/runs/1")
        self.client.post("/predict", json=REQUEST)
        self.assertEqual(hashlib.sha256(self.database.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
