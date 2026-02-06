import copy
from dataclasses import asdict, dataclass

from agentops.arg_configs import ProviderConfig, TestConfig
from agentops.llm_user.llm_user_v1 import LLMUser
from agentops.prompt.template_render import LlamaUserTemplateRenderer
from agentops.resource_map import ResourceMap
from agentops.runtime_adapter.wxo_runtime_adapter import WXORuntimeAdapter
from agentops.service_provider import get_provider
from agentops.service_provider.provider import Provider
from agentops.wxo_client import WXOClient, get_wxo_client


@dataclass
class Clients:
    wxo_client: WXOClient
    llmaaj_provider: Provider
    resource_map: ResourceMap
    inference_backend: WXORuntimeAdapter
    llm_user: LLMUser


def bootstrap_clients(config: TestConfig) -> Clients:
    """
    Bootstrap all clients needed for the evaluation.

    Args:
        config: The test configuration

    Returns:
        A tuple containing:
        - wxo_client: The WXO client
        - llmaaj_provider: The provider for custom metrics
        - resource_map: The resource map
        - inference_backend: The inference backend
        - llm_user: The LLM user
    """
    # Initialize WXO client
    wxo_client = get_wxo_client(
        config.auth_config.url,
        config.auth_config.tenant_name,
        config.auth_config.token,
    )

    # Initialize provider for custom metrics
    original_provider_config = config.provider_config
    provider_config_dict = asdict(original_provider_config)

    provider_kwargs = {
        "config": ProviderConfig(**provider_config_dict),
        "model_id": config.llm_user_config.model_id,
    }

    if provider_config_dict.get("provider", "gateway") == "gateway":
        provider_kwargs.update(
            token=config.auth_config.token or wxo_client.api_key,
            instance_url=wxo_client.service_url,
        )
        config.auth_config.token = (
            config.auth_config.token or wxo_client.api_key
        )
        config.auth_config.url = (
            config.auth_config.url or wxo_client.service_url
        )

    # Initialize resource map
    resource_map = ResourceMap(wxo_client)

    # Initialize inference backend
    inference_backend = WXORuntimeAdapter(wxo_client=wxo_client)

    # Initialize LLM user
    llm_user = LLMUser(
        wai_client=get_provider(**provider_kwargs),
        template=LlamaUserTemplateRenderer(
            config.llm_user_config.prompt_config
        ),
        user_response_style=config.llm_user_config.user_response_style,
    )

    llamaj_provider_kwargs = copy.deepcopy(provider_kwargs)
    llamaj_config_dict = asdict(llamaj_provider_kwargs["config"])

    # ============ GOVCLOUD FIX: Handle both dict and dataclass ============
    cmc = config.custom_metrics_config
    if isinstance(cmc, dict):
        llmaaj_cfg = cmc.get("llmaaj_config", {})
        llamaj_config_dict["model_id"] = llmaaj_cfg.get("model_id", llamaj_config_dict["model_id"])
        llamaj_config_dict["embedding_model_id"] = llmaaj_cfg.get("embedding_model_id", llamaj_config_dict["embedding_model_id"])
    else:
        llamaj_config_dict["model_id"] = cmc.llmaaj_config.model_id
        llamaj_config_dict["embedding_model_id"] = cmc.llmaaj_config.embedding_model_id
    # ============ END GOVCLOUD FIX ============

    llamaj_provider_kwargs["config"] = ProviderConfig(**llamaj_config_dict)
    llmaaj_provider = get_provider(**llamaj_provider_kwargs)

    return Clients(
        wxo_client=wxo_client,
        llmaaj_provider=llmaaj_provider,
        resource_map=resource_map,
        inference_backend=inference_backend,
        llm_user=llm_user,
    )
