#!/usr/bin/env python3
"""
FedRAMP/GovCloud Activation Script
===================================
Activates a GovCloud environment by exchanging API key for MCSP token.

Usage:
    python fedramp_activate.py <env-name> --api-key <your-api-key>
"""
from typing import Annotated
import typer
import requests
import os

from ibm_watsonx_orchestrate.cli.commands.tools.types import RegistryType
from ibm_watsonx_orchestrate.cli.config import (
    Config, AUTH_CONFIG_FILE_FOLDER, AUTH_CONFIG_FILE,
    ENV_WXO_URL_OPT, AUTH_SECTION_HEADER, AUTH_MCSP_TOKEN_OPT,
    PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT,
    PYTHON_REGISTRY_TEST_PACKAGE_VERSION_OVERRIDE_OPT,
    PYTHON_REGISTRY_SKIP_VERSION_CHECK_OPT,
    CONTEXT_SECTION_HEADER, CONTEXT_ACTIVE_ENV_OPT,
    ENVIRONMENTS_SECTION_HEADER, DEFAULT_CONFIG_FILE_CONTENT
)
from ibm_watsonx_orchestrate.client.utils import check_token_validity, is_local_dev

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False
)

# SSL Certificate path - update this to your certificate location
SSL_CERT_PATH = os.environ.get("SSL_CERT_FILE", "/path/to/your/certificate.pem")


@app.command(name='activate', no_args_is_help=True)
def activate(
        name: Annotated[
            str,
            typer.Argument(),
        ],
        apikey: Annotated[
            str,
            typer.Option(
                "--api-key", "-a", help="WXO API Key."
            ),
        ],
        registry: Annotated[
            RegistryType,
            typer.Option("--registry", help="Which registry to use when importing python tools", hidden=True),
        ] = None,
        test_package_version_override: Annotated[
            str,
            typer.Option("--test-package-version-override", help="Which prereleased package version to reference when using --registry testpypi", hidden=True),
        ] = None,
        skip_version_check: Annotated[
            bool,
            typer.Option('--skip-version-check/--enable-version-check', help='Use this flag to skip validating that adk version in use exists in pypi.')
        ] = None,
        skip_ssl_verify: Annotated[
            bool,
            typer.Option('--skip-ssl-verify', help='Skip SSL certificate verification (not recommended for production).')
        ] = False,
):
    cfg = Config()
    auth_cfg = Config(AUTH_CONFIG_FILE_FOLDER, AUTH_CONFIG_FILE)
    env_cfg = cfg.read(ENVIRONMENTS_SECTION_HEADER, name)
    url = cfg.get(ENVIRONMENTS_SECTION_HEADER, name, ENV_WXO_URL_OPT)
    is_local = is_local_dev(url)

    if not env_cfg:
        print(f"Environment '{name}' does not exist. Please create it with `orchestrate env add`")
        return
    elif not env_cfg.get(ENV_WXO_URL_OPT):
        print(f"Environment '{name}' is misconfigured. Please re-create it with `orchestrate env add`")
        return

    existing_auth_config = auth_cfg.get(AUTH_SECTION_HEADER).get(name, {})
    existing_token = existing_auth_config.get(AUTH_MCSP_TOKEN_OPT) if existing_auth_config else None

    if not check_token_validity(existing_token) or is_local:
        try:
            # Determine SSL verification
            verify = False if skip_ssl_verify else SSL_CERT_PATH
            if not skip_ssl_verify and not os.path.exists(SSL_CERT_PATH):
                print(f"Warning: SSL cert not found at {SSL_CERT_PATH}")
                print("Using --skip-ssl-verify or set SSL_CERT_FILE env var")
                verify = False

            resp = requests.post(
                "https://dai.ibmforusgov.com/api/rest/mcsp/apikeys/token",
                json={'apikey': apikey},
                verify=verify
            )
            resp.raise_for_status()
            resp = resp.json()
            auth_cfg.save(
                {
                    AUTH_SECTION_HEADER: {
                        name: {
                            'wxo_mcsp_token': resp['token'],
                            'wxo_mcsp_token_expiry': resp['expiration']
                        }
                    },
                }
            )
            print(f"Token obtained successfully!")
        except requests.exceptions.HTTPError as e:
            print(f"Error: {e.response.status_code} - {e.response.text}")
            exit(1)
        except Exception as e:
            print(f"Error: {e}")
            exit(1)

    cfg.write(CONTEXT_SECTION_HEADER, CONTEXT_ACTIVE_ENV_OPT, name)
    if registry is not None:
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT, str(registry))
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TEST_PACKAGE_VERSION_OVERRIDE_OPT, test_package_version_override)
    elif cfg.read(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT) is None:
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TYPE_OPT, DEFAULT_CONFIG_FILE_CONTENT[PYTHON_REGISTRY_HEADER][PYTHON_REGISTRY_TYPE_OPT])
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_TEST_PACKAGE_VERSION_OVERRIDE_OPT, test_package_version_override)
    if skip_version_check is not None:
        cfg.write(PYTHON_REGISTRY_HEADER, PYTHON_REGISTRY_SKIP_VERSION_CHECK_OPT, skip_version_check)

    print(f"Environment '{name}' is now active")


if __name__ == "__main__":
    app()
