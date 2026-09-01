#!/usr/bin/env bash
# Deploy DocMind to Azure Container Apps (Switzerland North) with a managed Postgres.
#
# Prerequisites: `az login`, an Azure subscription, and the Azure OpenAI keys you want to use.
# Everything is idempotent: re-running updates the existing resources.
#
#   ./deploy/azure/deploy.sh                # uses the defaults below
#   LOCATION=westeurope ./deploy/azure/deploy.sh
#
# Cost note (2026 list prices, approx.): Container App 2 vCPU / 4 GiB ≈ USD 60-70 / month if
# always on (scale-to-zero makes it near zero when idle, at the price of a ~2 min cold start
# for model download/load); Postgres Flexible Server B1ms ≈ USD 15 / month.
set -euo pipefail

: "${RG:=docmind-rg}"
: "${LOCATION:=switzerlandnorth}"
: "${ACR:=docmindacr$RANDOM}"          # must be globally unique, lower-case
: "${ENV_NAME:=docmind-env}"
: "${APP:=docmind-api}"
: "${PG:=docmind-pg-$RANDOM}"
: "${PG_USER:=docmind}"
: "${PG_PASSWORD:=$(openssl rand -base64 24 | tr -d '/+=')}"
: "${BASIC_AUTH_USER:=demo}"
: "${BASIC_AUTH_PASSWORD:=$(openssl rand -base64 12 | tr -d '/+=')}"
: "${LLM_BACKEND:=azure}"
: "${AZURE_OPENAI_API_KEY:?set AZURE_OPENAI_API_KEY}"
: "${AZURE_OPENAI_ENDPOINT:?set AZURE_OPENAI_ENDPOINT}"
: "${AZURE_OPENAI_DEPLOYMENT:=gpt-4o-mini}"

echo "== resource group"
az group create -n "$RG" -l "$LOCATION" -o none

echo "== container registry + image build (in the cloud, no local Docker needed)"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none
az acr build -r "$ACR" -t docmind-api:latest . -o none
ACR_LOGIN=$(az acr show -n "$ACR" --query loginServer -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)

echo "== Postgres Flexible Server with pgvector"
az postgres flexible-server create -n "$PG" -g "$RG" -l "$LOCATION" \
  --admin-user "$PG_USER" --admin-password "$PG_PASSWORD" \
  --sku-name Standard_B1ms --tier Burstable --storage-size 32 --version 16 \
  --database-name docmind --public-access 0.0.0.0 -o none
az postgres flexible-server parameter set -g "$RG" -s "$PG" \
  --name azure.extensions --value vector -o none
PG_HOST=$(az postgres flexible-server show -n "$PG" -g "$RG" --query fullyQualifiedDomainName -o tsv)
DATABASE_URL="postgresql+psycopg://$PG_USER:$PG_PASSWORD@$PG_HOST:5432/docmind?sslmode=require"

echo "== Container Apps environment + app"
az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" -o none
az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
  --image "$ACR_LOGIN/docmind-api:latest" \
  --registry-server "$ACR_LOGIN" --registry-username "$ACR" --registry-password "$ACR_PASS" \
  --target-port 8000 --ingress external \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 \
  --secrets "database-url=$DATABASE_URL" "aoai-key=$AZURE_OPENAI_API_KEY" \
            "basic-pass=$BASIC_AUTH_PASSWORD" \
  --env-vars "DATABASE_URL=secretref:database-url" \
             "LLM_BACKEND=$LLM_BACKEND" \
             "AZURE_OPENAI_API_KEY=secretref:aoai-key" \
             "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
             "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT" \
             "EMBEDDING_BACKEND=local" \
             "BASIC_AUTH_USER=$BASIC_AUTH_USER" \
             "BASIC_AUTH_PASSWORD=secretref:basic-pass" \
             "HF_HOME=/models" \
  -o none

URL="https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
cat <<EOF

Deployed.
  URL:            $URL          (login: $BASIC_AUTH_USER / $BASIC_AUTH_PASSWORD)
  Health:         $URL/health
  Postgres:       $PG_HOST  (user $PG_USER)

Next: ingest the corpus into the cloud database from your laptop —
  DATABASE_URL='$DATABASE_URL' uv run python -m docmind.ingest data/raw/
EOF
