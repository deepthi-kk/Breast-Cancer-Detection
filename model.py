import json

from marshmallow import Schema, fields


class PredictReq(Schema):
    radius_mean = fields.Float(required=True)
    perimeter_mean = fields.Float(required=True)
    area_mean = fields.Float(required=True)
    concavity_mean = fields.Float(required=True)
    concave_points_mean = fields.Float(required=True)
    radius_worst = fields.Float(required=True)
    perimeter_worst = fields.Float(required=True)
    area_worst = fields.Float(required=True)
    concavity_worst = fields.Float(required=True)
    concave_points_worst = fields.Float(required=True)


class PredictRes:
    output: str

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def __init__(self, output: str) -> None:
        self.output = output
