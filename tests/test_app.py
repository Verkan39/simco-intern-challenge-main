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
