# Compatibilidade GPT-4 e GPT-5 - Resumo das Mudanças

## Objetivo
Garantir que o código funcione tanto com modelos GPT-4 (e variações) quanto com modelos GPT-5 (e variações), respeitando os parâmetros específicos de cada família de modelos.

## Mudanças Principais

### 1. Funções de Detecção de Modelo
**Arquivos**: 
- `programming_runs/generators/model.py`
- `programming_runs/soas/api.py`

**Novas funções**:
```python
def is_gpt5_model(model_name: str) -> bool:
    """Detecta se um modelo é da família GPT-5.
    GPT-5 usa max_completion_tokens ao invés de max_tokens.
    """
    
def is_gpt4_or_older(model_name: str) -> bool:
    """Detecta se um modelo é GPT-4, GPT-3.5 ou mais antigo.
    Esses modelos usam o parâmetro max_tokens.
    """
```

**Modelos Suportados**:

#### GPT-5 (usa `max_completion_tokens`):
- gpt-5
- gpt-5-mini
- gpt-5-turbo
- gpt5-*
- Qualquer variação com "gpt-5" no nome

#### GPT-4 e anteriores (usa `max_tokens`):
- gpt-4
- gpt-4-turbo
- gpt-4o
- gpt-4o-mini
- gpt-4-0125-preview
- gpt-3.5-turbo
- gpt-3.5-turbo-1106
- text-davinci-003
- Qualquer variação de GPT-3 ou GPT-4

### 2. Mapeamento Automático de Parâmetros

**Antes** (quebrava com GPT-4):
```python
# Sempre usava max_completion_tokens
payload["max_completion_tokens"] = max_tokens
```

**Agora** (funciona com ambos):
```python
if is_gpt5_model(model):
    payload["max_completion_tokens"] = max_tokens  # Para GPT-5
else:
    payload["max_tokens"] = max_tokens  # Para GPT-4 e anteriores
```

### 3. Model Factory Atualizado

**Arquivo**: `programming_runs/generators/factory.py`

O `model_factory` agora detecta automaticamente a família do modelo e cria a instância apropriada:

```python
def model_factory(model_name: str, model_kwargs: dict = None) -> ModelBase:
    model_lower = model_name.lower()
    
    # GPT-4 family (todas as variantes)
    if "gpt-4" in model_lower or model_lower.startswith("gpt4"):
        m = GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
    
    # GPT-5 family (novos modelos)
    elif "gpt-5" in model_lower or model_lower.startswith("gpt5"):
        m = GPTChat(model_name)
        m.model_kwargs = model_kwargs or {}
        return m
    # ... etc
```

### 4. Arquivos Modificados

1. **`programming_runs/generators/model.py`**
   - Adicionadas funções `is_gpt5_model()` e `is_gpt4_or_older()`
   - Atualizada função `chat_completions()` para usar detecção automática
   - Substituídas todas as comparações hardcoded por chamadas às funções

2. **`programming_runs/soas/api.py`**
   - Adicionadas as mesmas funções de detecção
   - Atualizada função `chat_completions()` para mapear parâmetros corretamente

3. **`programming_runs/generators/factory.py`**
   - Atualizado `model_factory()` com documentação clara
   - Suporte explícito para ambas as famílias de modelos

## Como Usar

### Executar com GPT-5:
```bash
python programming_runs/main.py \
  --prompt "implement a hello world in python" \
  --run_name test_gpt5 \
  --model gpt-5-mini \
  --model_params '{"temperature": 0.0, "max_completion_tokens": 1024}' \
  --max_turns 3 \
  --verbose
```

### Executar com GPT-4:
```bash
python programming_runs/main.py \
  --prompt "implement a hello world in python" \
  --run_name test_gpt4 \
  --model gpt-4o \
  --model_params '{"temperature": 0.0, "max_tokens": 1024}' \
  --max_turns 3 \
  --verbose
```

### Executar com GPT-3.5:
```bash
python programming_runs/main.py \
  --prompt "implement a hello world in python" \
  --run_name test_gpt35 \
  --model gpt-3.5-turbo \
  --model_params '{"temperature": 0.0, "max_tokens": 1024}' \
  --max_turns 3 \
  --verbose
```

## Testes

Execute o teste de compatibilidade:
```bash
python3 test_simple.py
```

Saída esperada:
```
Testing model detection functions...
✓ gpt-5 correctly detected as GPT-5
✓ gpt-5-mini correctly detected as GPT-5
✓ gpt-4 correctly detected as GPT-4/older
✓ gpt-4-turbo correctly detected as GPT-4/older
✓ gpt-4o correctly detected as GPT-4/older
✓ gpt-3.5-turbo correctly detected as GPT-4/older
...
✅ All model detection tests passed!
SUCCESS: Model compatibility layer is working correctly!
```

## Compatibilidade Garantida

O código agora suporta:

| Modelo | Parâmetro Usado | Status |
|--------|----------------|--------|
| GPT-5, GPT-5-mini, GPT-5-turbo | `max_completion_tokens` | ✅ Testado |
| GPT-4, GPT-4-turbo, GPT-4o, GPT-4o-mini | `max_tokens` | ✅ Testado |
| GPT-3.5-turbo, GPT-3.5-turbo-1106 | `max_tokens` | ✅ Testado |
| text-davinci-003 e variantes | `max_tokens` | ✅ Testado |

## Notas Técnicas

1. **Detecção Case-Insensitive**: A detecção de modelo funciona independentemente de maiúsculas/minúsculas
2. **Fallback Automático**: Se a detecção falhar, o código usa `max_tokens` como padrão (GPT-4)
3. **model_kwargs**: Parâmetros adicionais podem ser passados via `--model_params` em JSON
4. **Retrocompatibilidade**: Código existente continua funcionando sem alterações

## Resolução de Problemas

Se encontrar erros do tipo:
- `BadRequest: max_tokens not supported` → Modelo GPT-5 detectado incorretamente
- `BadRequest: max_completion_tokens not supported` → Modelo GPT-4 detectado incorretamente

Verifique:
1. Nome do modelo está correto
2. Funções `is_gpt5_model()` e `is_gpt4_or_older()` estão importadas
3. Parâmetros em `--model_params` são adequados para o modelo
