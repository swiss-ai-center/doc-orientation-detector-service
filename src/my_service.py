from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from flipml import flip_detector as fd


api_description = """The doc orientation detector service detects if a scanned document is upside down.
It returns the angle of rotation, which can be either 0 or 180 degrees.
"""
api_summary = """Detects if a scanned document is upside down.
"""
api_title = "Document Orientation Detector API."
version = "0.0.1"


settings = get_settings()


class MyService(Service):
    """
    flipml model - detects if a scanned document is upside down
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Document Orientation Detector",
            slug="doc-orientation-detector",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.TEXT_PLAIN]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/doc-orientation-detector/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type

        raw = data["image"].data
        # input_type = data["image"].type
        img_dim = (512, 512)

        model = fd.load_model()
        image = BytesIO(raw)
        pil_image = Image.open(image)
        pil_image = ImageOps.grayscale(pil_image)

        # TODO: improve resizing
        data = np.array(pil_image.resize(img_dim))
        data = np.expand_dims(data, axis=0)  # (1, 512, 512) we need a batch
        rotation = fd.predict(model, data)
        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields

        output = str(rotation)
        return {
            "result": TaskData(data=output, type=FieldDescriptionType.TEXT_PLAIN)
        }
