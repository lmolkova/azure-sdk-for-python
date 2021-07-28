# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import functools
from typing import TYPE_CHECKING, overload
import warnings

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    map_error,
)
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest

from .. import models as _models, _rest as rest

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from typing import Any, Callable, Dict, Generic, Optional, TypeVar

    T = TypeVar("T")
    ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]


class QuestionAnsweringClientOperationsMixin(object):
    @overload
    def query_knowledgebase(
        self,
        knowledge_base_query_options,  # type: "_models.KnowledgeBaseQueryOptions"
        **kwargs  # type: Any
    ):
        # type: (...) -> "_models.KnowledgeBaseAnswers"
        """Answers the specified question using your knowledge base.

        :param knowledge_base_query_options: Post body of the request.
        :type knowledge_base_query_options:
         ~azure.ai.language.questionanswering.models.KnowledgeBaseQueryOptions
        :keyword project_name: The name of the project to use.
        :paramtype project_name: str
        :keyword deployment_name: The name of the specific deployment of the project to use.
        :paramtype deployment_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: KnowledgeBaseAnswers, or the result of cls(response)
        :rtype: ~azure.ai.language.questionanswering.models.KnowledgeBaseAnswers
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        pass

    @overload
    def query_knowledgebase(
        self,
        **kwargs  # type: Any
    ):
        # type: (...) -> "_models.KnowledgeBaseAnswers"
        """Answers the specified question using your knowledge base.

        :keyword project_name: The name of the project to use.
        :paramtype project_name: str
        :keyword deployment_name: The name of the specific deployment of the project to use.
        :paramtype deployment_name: str
        :keyword question: User question to query against the knowledge base.
        :paramtype question: str
        :keyword qna_id: Exact QnA ID to fetch from the knowledge base, this field takes priority over
        question.
        :paramtype qna_id: int
        :keyword top: Max number of answers to be returned for the question.
        :paramtype top: int
        :keyword user_id: Unique identifier for the user.
        :paramtype user_id: str
        :keyword confidence_score_threshold: Minimum threshold score for answers, value ranges from 0 to
        1.
        :paramtype confidence_score_threshold: float
        :keyword context: Context object with previous QnA's information.
        :paramtype context: ~azure.ai.language.questionanswering.models.KnowledgeBaseAnswerRequestContext
        :keyword ranker_type: (Optional) Set to 'QuestionOnly' for using a question only Ranker. Possible
        values include: "Default", "QuestionOnly".
        :paramtype ranker_type: str or ~azure.ai.language.questionanswering.models.RankerType
        :keyword strict_filters: Filter QnAs based on give metadata list and knowledge base source names.
        :paramtype strict_filters: ~azure.ai.language.questionanswering.models.StrictFilters
        :keyword answer_span_request: To configure Answer span prediction feature.
        :paramtype answer_span_request: ~azure.ai.language.questionanswering.models.AnswerSpanRequest
        :keyword include_unstructured_sources: (Optional) Flag to enable Query over Unstructured Sources.
        :paramtype include_unstructured_sources: bool
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: KnowledgeBaseAnswers, or the result of cls(response)
        :rtype: ~azure.ai.language.questionanswering.models.KnowledgeBaseAnswers
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        pass

    def query_knowledgebase(
        self,
        *args,  # type: "_models.KnowledgeBaseQueryOptions"
        **kwargs  # type: Any
    ):
        # type: (...) -> "_models.KnowledgeBaseAnswers"
        """Answers the specified question using your knowledge base.

        :param knowledge_base_query_options: Post body of the request. Provide either `knowledge_base_query_options`, OR
         individual keyword arguments. If both are provided, only the options object will be used.
        :type knowledge_base_query_options:
         ~azure.ai.language.questionanswering.models.KnowledgeBaseQueryOptions
        :keyword project_name: The name of the project to use.
        :paramtype project_name: str
        :keyword deployment_name: The name of the specific deployment of the project to use.
        :paramtype deployment_name: str
        :keyword question: User question to query against the knowledge base. Provide either `knowledge_base_query_options`, OR
         individual keyword arguments. If both are provided, only the options object will be used.
        :paramtype question: str
        :keyword qna_id: Exact QnA ID to fetch from the knowledge base, this field takes priority over question.
        :paramtype qna_id: int
        :keyword top: Max number of answers to be returned for the question.
        :paramtype top: int
        :keyword user_id: Unique identifier for the user.
        :paramtype user_id: str
        :keyword confidence_score_threshold: Minimum threshold score for answers, value ranges from 0 to 1.
        :paramtype confidence_score_threshold: float
        :keyword context: Context object with previous QnA's information.
        :paramtype context: ~azure.ai.language.questionanswering.models.KnowledgeBaseAnswerRequestContext
        :keyword ranker_type: (Optional) Set to 'QuestionOnly' for using a question only Ranker. Possible
         values include: "Default", "QuestionOnly".
        :paramtype ranker_type: str or ~azure.ai.language.questionanswering.models.RankerType
        :keyword strict_filters: Filter QnAs based on give metadata list and knowledge base source names.
        :paramtype strict_filters: ~azure.ai.language.questionanswering.models.StrictFilters
        :keyword answer_span_request: To configure Answer span prediction feature.
        :paramtype answer_span_request: ~azure.ai.language.questionanswering.models.AnswerSpanRequest
        :keyword include_unstructured_sources: (Optional) Flag to enable Query over Unstructured Sources.
        :paramtype include_unstructured_sources: bool
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: KnowledgeBaseAnswers, or the result of cls(response)
        :rtype: ~azure.ai.language.questionanswering.models.KnowledgeBaseAnswers
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        if args:
            knowledge_base_query_options = args[0]
        else:
            knowledge_base_query_options = _models.KnowledgeBaseQueryOptions(
                qna_id=kwargs.pop("qna_id", None),
                question=kwargs.pop("question", None),
                top=kwargs.pop("top", None),
                user_id=kwargs.pop("user_id", None),
                confidence_score_threshold=kwargs.pop("confidence_score_threshold", None),
                context=kwargs.pop("context", None),
                ranker_type=kwargs.pop("ranker_type", None),
                strict_filters=kwargs.pop("strict_filters", None),
                answer_span_request=kwargs.pop("answer_span_request", None),
                include_unstructured_sources=kwargs.pop("include_unstructured_sources", None)
            )
        cls = kwargs.pop("cls", None)  # type: ClsType["_models.KnowledgeBaseAnswers"]
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop("error_map", {}))
        content_type = kwargs.pop("content_type", "application/json")  # type: Optional[str]
        project_name = kwargs.pop("project_name")  # type: str
        deployment_name = kwargs.pop("deployment_name", None)  # type: Optional[str]

        json = self._serialize.body(knowledge_base_query_options, "KnowledgeBaseQueryOptions")

        request = rest.build_query_knowledgebase_request(
            content_type=content_type,
            project_name=project_name,
            deployment_name=deployment_name,
            json=json,
            template_url=self.query_knowledgebase.metadata["url"],
        )._to_pipeline_transport_request()
        path_format_arguments = {
            "Endpoint": self._serialize.url("self._config.endpoint", self._config.endpoint, "str", skip_quote=True),
        }
        request.url = self._client.format_url(request.url, **path_format_arguments)

        pipeline_response = self._client.send_request(request, stream=False, _return_pipeline_response=True, **kwargs)
        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error)

        deserialized = self._deserialize("KnowledgeBaseAnswers", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    query_knowledgebase.metadata = {"url": "/:query-knowledgebases"}  # type: ignore

    @overload
    def query_text(
        self,
        text_query_options,  # type: "_models.TextQueryOptions"
        **kwargs  # type: Any
    ):
        # type: (...) -> "_models.TextAnswers"
        """Answers the specified question using the provided text in the body.

        :param text_query_options: Post body of the request.
        :type text_query_options: ~azure.ai.language.questionanswering.models.TextQueryOptions
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: TextAnswers, or the result of cls(response)
        :rtype: ~azure.ai.language.questionanswering.models.TextAnswers
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        pass

    @overload
    def query_text(
        self,
        **kwargs  # type: Any
    ):
        # type: (...) -> "_models.TextAnswers"
        """Answers the specified question using the provided text in the body.

        :keyword question: Required. User question to query against the given text records.
        :paramtype question: str
        :keyword records: Required. Text records to be searched for given question.
        :paramtype records: list[~azure.ai.language.questionanswering.models.TextInput]
        :keyword language: Language of the text records. This is BCP-47 representation of a language. For
        example, use "en" for English; "es" for Spanish etc. If not set, use "en" for English as
        default.
        :paramtype language: str
        :keyword string_index_type: Specifies the method used to interpret string offsets.  Defaults to
        Text Elements (Graphemes) according to Unicode v8.0.0. For additional information see
        https://aka.ms/text-analytics-offsets. Possible values include: "TextElements_v8",
        "UnicodeCodePoint", "Utf16CodeUnit". Default value: "TextElements_v8".
        :paramtype string_index_type: str or ~azure.ai.language.questionanswering.models.StringIndexType
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: TextAnswers, or the result of cls(response)
        :rtype: ~azure.ai.language.questionanswering.models.TextAnswers
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        pass

    def query_text(
        self,
        *args,  # type: "_models.TextQueryOptions"
        **kwargs  # type: Any
    ):
        # type: (...) -> "_models.TextAnswers"
        """Answers the specified question using the provided text in the body.

        :param text_query_options: Post body of the request. Provide either `text_query_options`, OR
         individual keyword arguments. If both are provided, only the options object will be used.
        :type text_query_options: ~azure.ai.language.questionanswering.models.TextQueryOptions
        :keyword question: User question to query against the given text records. Provide either `text_query_options`, 
         OR individual keyword arguments. If both are provided, only the options object will be used.
        :paramtype question: str
        :keyword records: Text records to be searched for given question. Provide either `text_query_options`, OR
         individual keyword arguments. If both are provided, only the options object will be used.
        :paramtype records: list[~azure.ai.language.questionanswering.models.TextInput]
        :keyword language: Language of the text records. This is BCP-47 representation of a language. For
         example, use "en" for English; "es" for Spanish etc. If not set, use "en" for English as default.
        :paramtype language: str
        :keyword string_index_type: Specifies the method used to interpret string offsets.  Defaults to
         Text Elements (Graphemes) according to Unicode v8.0.0. For additional information see
         https://aka.ms/text-analytics-offsets. Possible values include: "TextElements_v8",
         "UnicodeCodePoint", "Utf16CodeUnit". Default value: "TextElements_v8".
        :paramtype string_index_type: str or ~azure.ai.language.questionanswering.models.StringIndexType
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: TextAnswers, or the result of cls(response)
        :rtype: ~azure.ai.language.questionanswering.models.TextAnswers
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        if args:
            text_query_options = args[0]
        else:
            text_query_options = _models.TextQueryOptions(
                question=kwargs.pop("question"),
                records=kwargs.pop("records"),
                language=kwargs.pop("language", None),
                string_index_type=kwargs.pop("string_index_type", "TextElements_v8")
            )
        cls = kwargs.pop("cls", None)  # type: ClsType["_models.TextAnswers"]
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError}
        error_map.update(kwargs.pop("error_map", {}))
        content_type = kwargs.pop("content_type", "application/json")  # type: Optional[str]

        json = self._serialize.body(text_query_options, "TextQueryOptions")

        request = rest.build_query_text_request(
            content_type=content_type,
            json=json,
            template_url=self.query_text.metadata["url"],
        )._to_pipeline_transport_request()
        path_format_arguments = {
            "Endpoint": self._serialize.url("self._config.endpoint", self._config.endpoint, "str", skip_quote=True),
        }
        request.url = self._client.format_url(request.url, **path_format_arguments)

        pipeline_response = self._client.send_request(request, stream=False, _return_pipeline_response=True, **kwargs)
        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.ErrorResponse, response)
            raise HttpResponseError(response=response, model=error)

        deserialized = self._deserialize("TextAnswers", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    query_text.metadata = {"url": "/:query-text"}  # type: ignore
