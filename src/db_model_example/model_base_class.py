from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar
from uuid import UUID, uuid4

from bson import ObjectId
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic.alias_generators import to_camel
from pydantic.types import StringConstraints
from pydantic_core import core_schema

from ffcs_data_model.ffcs_enums import (
    AllowedDataProcessing,
    AllowedGroupAnalysisPipeline,
    AllowedImageFileType,
    AllowedLigand,
    AllowedLightsources,
    AllowedRefinement,
    CoordinateFileType,
    ReflectionFileType,
    ReindexOperator,
    SpaceGroupNameChiral,
)

M = TypeVar("M", bound="AnalysisBase")

SCHEMAVERSION = "0.1.0"


def to_bson_value(value: Any) -> Any:
    """Convert Python objects to BSON-compatible values for MongoDB.
    Handles Path, timedelta, UUID, and nested structures (Mappings, e.g. dicts, lists, tuples).
    for now keep UUIDs as string (usecase for non "_id"-key associated UUID values. "_id" should be treated as ObjectId, s.a.)
    consider uuidRepresentation="standard" for UUID, might require shelling if mixture of UUID and string.

    Args:
        value: Python value to convert.

    Returns:
        converted, BSON-compatible value.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Mapping):
        return {
            key: to_bson_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_bson_value(item)
            for item in value
        ]

    return value


class AnalysisBase(BaseModel):
    """
    common behavior inherited by all models in Analysis/Processing.
    Includes type translation for the different transfer modes:
    - json compatible dicts for http requests (.to_http_dict())
    - BSON compatible dicts for MongoDB insertion (.to_db_dict())
    - validated model object with rich python types after retrieving BSON compatibles from DB (.from_db())
    
    use python native snake case for python dicts in application code and Pydantic inherent
    to_camel alias generator at http transfer and for database insetion
    use .model_validate()/.from_db() to get back snake case and python types

    
    """
    model_config = ConfigDict(
        extra = "forbid", # reject unknown keys
        alias_generator=to_camel,
        # populate_by_name = True, # accept alias or field names, depreciated now, replaced by vbn, vba
        # use_enum_values = True, # serialise enums as their values,might be useful, decide later. having the enums might be beneficial
        validate_by_name=True, #accept python snake case
        validate_by_alias=True, #accept Mongo Camel case
        validate_assignment=True, #revalidate values when they are changed after object creation and initial validation
        validate_default=True, # also validate default values
        validation_error_cause = True, # more verbose VE, include underlying exception causes in validation diagnostics
        str_strip_whitespace = True,
        #serialize_by_alias=True  #is this a good idea? probably not, the helper mmethods handle this already explicitly
    )

    # ---------- HTTP / JSON boundary ----------

    def to_http_dict(
        self,
        *, # everything after the "*"" must be passed as keyword=value pair, not just "value, value,.."
        by_alias: bool = True,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert the object to JSON-compatibly typed dict
        For requests/FastAPI payloads.
        Ensures UUID/datetime/Path/Enums etc. are serialized to JSON compatible representation.
        use to send data to fastAPI server via http request
        datetime becomes string!

        Args:
            by_alias: use pydantic field aliases as keys
            exclude_none: omit fields with value equal None
            exclude_unset: omit fields not set explicitly
            exclude_defaults: omit fields containing their default value
            **kwargs: additional options from `model_dump()`, see:
            https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/#pydantic.BaseModel.model_dump

        Returns:
            dict, json compatible but not(!) json string

        Examples:
            requests.post(url, json=payload.to_http_dict(**kwargs))
            useful things for kwargs are e.g. :
            include={"field_a", "field_b"}
            exclude={"secret_field"}
            context={...}
            warnings="error"
            round_trip=True
        """
        return self.model_dump(
            mode="json",
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            **kwargs,
        )


    def to_db_dict(
        self,
        *,
        by_alias: bool = True,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert the model BSON compatible dict
        !CAVE! USE to_db_insert_doc for new insertions! CAVE!
        unless you know what you are doing
        return a BSON oriented, "DB ready" dict for insertion into MongoDB via PyMongo client
        Keeps Python/Mongo-native objects intact (ObjectId, datetime, etc.).
        Use for:
        - replace entires while keeping identity intact
        - diagnostics
        - potentially migration
        datetime stays datetime

        Args:
            by_alias: use pydantic field aliases as keys
            exclude_none: omit fields with value equal None
            exclude_unset: omit fields not set explicitly
            exclude_defaults: omit fields containing their default value
            **kwargs: additional options from `model_dump()`

        Returns:
            dict, bson compatible values

        Examples:
            USE ONLY IN to_db_insert_doc() for insertion into MongoDB unless really sure you know what you are doing!
            collection.insert_one(data.to_db_insert_doc()) (.to_db_dict() if sure)
        """
        document = self.model_dump(
            mode="python", # retain the good python native types like datetime
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            **kwargs,
        )
        return to_bson_value(document) # now apply MongoDB storage choices, Path/UUID to string, etc.


    def to_db_insert_doc(self) -> dict[str, Any]:
        """Create a BSON compatible document for MongoDB insertion
        additional layer used afterto_db_dict() to remove "_id" if present, so MongoDB can set it itself
        explicit "id" is now allowed to be used as key name if desired

        Returns:
            dict, BSON compatible, "_id" field removed if present before
        """
        doc = self.to_db_dict()
        
        doc.pop("_id", None)
        return doc


    @classmethod
    def from_db(cls: type[M], doc: dict[str, Any]) -> M:
        """Create and validate a model object coming from  MongoDB
        matches dictionary keys to model field names OR aliases (both work)

        Args:
            doc: document from MongoDB

        Returns:
            validated instance of the class, e.g. SweepRecord, PreparedSampleRecord, etc.

        Raises:
            pydantic.ValidationError: if the document does not match model schema

        Examples:
            record = SweepRecord.from_db(doc)
        """
        return cls.model_validate(doc)

    