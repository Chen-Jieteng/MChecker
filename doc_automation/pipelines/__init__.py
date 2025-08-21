from dagster import Definitions

from .doc_generation_pipeline import doc_generation_assets, doc_generation_job
from .data_refresh_pipeline import data_refresh_assets, data_refresh_job
from .validation_pipeline import validation_assets, validation_job

defs = Definitions(
    assets=[
        *doc_generation_assets,
        *data_refresh_assets,
        *validation_assets
    ],
    jobs=[
        doc_generation_job,
        data_refresh_job,
        validation_job
    ]
)
