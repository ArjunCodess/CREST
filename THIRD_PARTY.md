# Attributed external components

CREST combines its independently developed graph/temporal/wave models and OD
solver with the public models below. We do not claim authorship of those models.

* Organizer utilities: jacky850/trafficflowbench-public, MIT. Existing notice is
  preserved under starter/LICENSE.
  Vendored revision: f9e3fbc35c066419726987e470448f33d4c72fd9.
* anubisyy: TrafficFlowBench 0.86517 Public Assets, CC0-1.0 according to primary
  Kaggle metadata in external/anubisyy/dataset-metadata.json. Source and trained
  CatBoost/LightGBM checkpoints are under external/anubisyy. Source URL:
  https://www.kaggle.com/datasets/anubisyy/trafficflowbench-086517-public-assets
* Izbassar Orynbassar and Sevgi Erdogan, team BizAI: MIT code and U-Net2 weights,
  https://github.com/IzbassarO/tfbench-bizai-release, revision
  7ad3632d77a8393d715a7ca7ad314f0277ccbbd9. MIT notice is preserved under
  external/bizai/LICENSE. Only the U-Net2 checkpoint member is used here.

Every external prediction is regenerated from model weights and the participant
release. No supplied final submission or shipped prediction component is used.
No happyc0der code is included because its license remains unresolved. Raw
competition observations are excluded and must be supplied by the verifier.
