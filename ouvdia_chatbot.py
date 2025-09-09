from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Optional, Union, Dict, Iterator
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.outputs import RequestOutput
from vllm import SamplingParams
from vllm.utils import Counter
from huggingface_hub import login
from dotenv import load_dotenv
import gradio as gr
import logging, os
from pathlib import Path
from datetime import datetime
data_hora_atual = datetime.now().strftime("%Y-%m-%d")

# ========== Configuração do logging ==========

# Diretório para os logs
diretorio = Path('/ouvdia/logs')
diretorio.mkdir(parents=True, exist_ok=True)

# Nome do arquivo de log com data e hora
data_hora_atual = datetime.now().strftime("%Y-%m-%d")
arquivo_logs = f'ouvdia_logs_{data_hora_atual}.log'
# Caminho completo para o arquivo de log
caminho_arquivo_logs = diretorio / arquivo_logs

# Configuração do logging
logging.basicConfig(
    filename=caminho_arquivo_logs,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ========== Definição das classes ==========

class ModeloLinguagemContínuo:
    """
    Classe para inicializar e gerenciar um modelo de linguagem de fluxo contínuo.
    """

    def __init__(
        self,
        modelo: str,
        tipo_dados: str = "auto",
        quantizacao: Optional[str] = None,
        **kwargs,
    ) -> None:

        argumentos_motor = EngineArgs(
            model=modelo,
            quantization=quantizacao,
            dtype=tipo_dados,
            enforce_eager=True,  
            device="cuda",
        )
        self.llm_motor = LLMEngine.from_engine_args(
            argumentos_motor, 
            usage_context=UsageContext.LLM_CLASS
        )

        self.contador_requisicoes = Counter()

    def gerar(
        self,
        prompt: str,
        parametros_amostragem: Optional[SamplingParams] = None
    ) -> Iterator[RequestOutput]:
        """Gera a resposta do modelo de forma iterativa (stream)."""
        # Gera um novo ID de requisição
        id_requisicao = str(next(self.contador_requisicoes))
        # Adiciona a requisição ao motor
        self.llm_motor.add_request(id_requisicao, prompt, parametros_amostragem)

        # Continua processando até que todas as requisições sejam concluídas
        while self.llm_motor.has_unfinished_requests():
            saidas_etapa = self.llm_motor.step()
            for saida in saidas_etapa:
                yield saida


class InterfaceUsuario:
    """
    Classe que gerencia a interface do usuário via Gradio.
    """

    def __init__(
        self,
        llm: ModeloLinguagemContínuo,
        tokenizador: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        parametros_amostragem: Optional[SamplingParams] = None,
        system_prompt: str = (
            "Seu nome é OuvdIA, você é a Assistente Inteligente da Ouvidoria da Receita Federal do Brasil."
            "Você foi desenvolvida por um equipe multidisciplinar composta por servidores da área de negócios e especialistas em Inteligência Artifical."
            "Você fornece respostas concisas, precisas e bem estruturadas."
            "Nas suas respostas garanta clareza, exatidão jurídica e conformidade com a legislação vigente, especialmente a LGPD,"
            "bem como o uso ético e responsável de Inteligência Artifical."
	    "Você utiliza princípios éticos, incluindo transparência, responsabilidade, justiça, privacidade e segurança."
            "Seu tom é profissional e respeitoso."
            "Use a linguagem simples e inclusiva para garantir a acessibilidade e a compreensão das informações fornecidas."
            #"Foque sua resposta na última mensagem do usuário\n"
        )
    ) -> None:
        """
        Args:
            llm (ModeloLinguagemContínuo): Instância do modelo de linguagem.
            tokenizador (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Tokenizador pré-treinado.
            parametros_amostragem (Optional[SamplingParams]): Parâmetros de amostragem.
            system_prompt (str): Prompt inicial (role=system).
        """
        self.llm = llm
        self.tokenizador = tokenizador
        self.parametros_amostragem = parametros_amostragem
        self.system_prompt = system_prompt

    def _gerar(
        self,
        history: List[Dict[str, str]],
        prompt_usuario: str,
        texto_principal: str,
        contexto: str
    ) -> Iterator[List[Dict[str, str]]]:
        """
        Função chamada quando o usuário envia uma nova mensagem.
        Recebe o prompt do usuário, contexto, texto_principal e o histórico atual (lista de dicts).
        Retorna (via yield) o histórico atualizado em streaming.
        """

        # Arquivando logs da mensagem do usuário e resposta do modelo
        interacao_usuario = {
            'prompt_sistema' : self.system_prompt,
            'prompt_usuario' : prompt_usuario,
            'texto_principal' : texto_principal,
            'contexto' : contexto}
        logging.info(f'interacao_usuario|{interacao_usuario}')

        # Se houver contexto, adicionamos ao prompt com instruções obrigatórias
        if contexto:
            contexto = (
                ' Instruções obrigatórias para você: Responda com base exclusivamente no contexto fornecido abaixo '
                'e evite informações adicionais ou juízo de valor. Segue o contexto obrigatório: ' + contexto
            )

        # Se não houver histórico, iniciamos "system message" com o prompt padrão
        if not history:
            history = [
                {"role": "system", "content": self.system_prompt + prompt_usuario + contexto}
            ]
        # Havendo histórico, apenas adicionamos o prompt do usuário
        else:
            history.append({"role": "system", "content": prompt_usuario})

        # Adiciona a mensagem do usuário ao histórico
        history.append({"role": "user", "content": texto_principal})

        # Emite imediatamente o histórico, já com os prompts
        yield history

        # Cria um dicionário para a resposta do assistente
        response = {"role": "assistant", "content": ""}

        # Define a mensagem inicial para o modelo
        mensagem_inicial = {
            "role": "system",
            "content": self.system_prompt + prompt_usuario + contexto
        }

        # Define a mensagem do usuário
        mensagem_usuario = {
            "role": "user",
            "content": texto_principal
        }

        # Constrói o prompt final concatenando todas as mensagens
        prompt_para_modelo = self.tokenizador.apply_chat_template(
            [mensagem_inicial, mensagem_usuario], 
            tokenize=False
        )

        # Geração em streaming
        for chunk in self.llm.gerar(prompt_para_modelo, self.parametros_amostragem):
            # Pega o texto parcial do chunk e adiciona à resposta
            texto_chunk = chunk.outputs[0].text
            response["content"] = texto_chunk

            # Emite (yield) o histórico + a resposta parcial do assistente
            yield history + [response]

    # Funções para aplicar 'Gostei', 'Não Gostei', 'Desfazer' e 'Refazer' na interface
    def aplicar_gostei(self, data: gr.LikeData):
        """
        Função para capturar feedback de 'Gostei' ou 'Não Gostei' na interface do usuário.
        """
        if data.liked:
            print("like ", data.value)
        else:
            print("dislike ", data.value)

    def aplicar_desfazer(self, history, undo_data: gr.UndoData):
        """
        Função para desfazer a última interação no histórico do chatbot.
        """
        return history[:undo_data.index], history[undo_data.index]['content']

    def aplicar_refazer(self, history, prompt_usuario, texto_principal, contexto):
        """
        Função para refazer uma interação específica no histórico do chatbot.
        """
        historico_anterior = history[:-1]  # Refazer a partir da penúltima interação
        ultimo_prompt = history[-1]['content']

        # Gera novamente a resposta usando o prompt anterior
        yield from self._gerar(historico_anterior, ultimo_prompt, texto_principal, contexto)

    # Função para processar o feedback dos likes no Chatbot
    def processar_feedback(self, mensagens, like_data: gr.LikeData, estado):
        """
        Processa o feedback do usuário no Chatbot e retorna informações detalhadas.
        """
        # Captura os detalhes do feedback
        resposta_avaliada = like_data.value
        indice_mensagem = like_data.index
        curtiu = like_data.liked

        # Dados processados retornados como JSON
        dados_feedback = {
            "mensagens": mensagens,
            "resposta_avaliada": resposta_avaliada,
            "indice_mensagem": indice_mensagem,
            "curtiu": curtiu,
            **estado  # Incorpora as chaves e valores do estado diretamente
        }

        # Log do feedback
        logging.info(f"avaliacao|{dados_feedback}")

        #return dados_feedback    

    # Função para atualizar o estado do usuário
    def atualizar_estado(self, request: gr.Request, estado):
        """
        Atualiza o estado local com informações do usuário e ip.
        """
        ip_cliente = request.client.host
        usuario = request.username or "Desconhecido"

        # Atualização do estado
        estado["client_ip"] = ip_cliente
        estado["usuario_logado"] = usuario

        # Armazenando logging de usuário
        dict_estado = {'usuario':usuario, 'ip':ip_cliente}        
        logging.info(f'logging_usuario|{dict_estado}')

        return None, None

    def iniciar(self):
        """
        Configura e lança a interface Gradio.
        """

        # ------ TEMA E CSS DA INTERFACE (idênticos ao seu código) ------
        tema = gr.themes.Default(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="slate",
            font=["Arial", "sans-serif"],
            text_size="md",
        )

        estilo = """
        body {
            font-weight: bold;
        }
        .gr-markdown {
            font-weight: bold;
        }
        .gr-button {
            font-weight: bold;
        }
        .gr-textbox {
            font-weight: bold;
        }
        input[type="text"]::placeholder {
            color: gray;
            font-style: italic;
            content: 'Seu nome de usuário aqui'
        }
        input[type="password"]::placeholder {
            color: gray;
            font-style: italic;
            content: 'Sua senha aqui'
        }      
        """

        with gr.Blocks(theme=tema, css=estilo) as interface_chat:

            with gr.Row():
                gr.Image(
                    value='/ouvdia/assets/ouvdia_logo.jpeg',
                    type="filepath",
                    label='OuvdIA',
                    show_label=False,
                    interactive=False,
                    elem_id="image-1"
                )
                gr.Image(
                    value="/ouvdia/assets/grafos_capa_3.jpeg",
                    type="filepath",
                    label="Assistente Inteligente",
                    show_label=False,
                    interactive=False,                    
                    elem_id="image-2"
                )

            # Ajustes de CSS
            interface_chat.css = """
            #image-1 {
                flex: 1; /* Ocupa 1 unidade da linha */
                max-width: 14%; /* Garante que ocupa esse % da linha */
                height: auto; /* Mantém a proporção */
            }
            #image-2 {
                flex: 4; /* Ocupa 4 unidades da linha */
                max-width: 86%; /* Garante que ocupa esse % da linha */
                height: auto; /* Mantém a proporção */
            }
            """

            gr.Markdown("# 💡OuvdIA - Assistente Inteligente da Ouvidoria da RFB")

            with gr.Row():
                prompt_usuario = gr.Textbox(
                    label='1). Prompt (preenchimento obrigatório): Instruções para a OuvdIA',
                    placeholder='📝 Insira aqui as instruções...'
                )

                texto_principal = gr.Textbox(
                    label='2). Texto Principal (preenchimento obrigatório)',
                    lines=1,  # Permite dar Enter e enviar
                    placeholder='📝 Insira aqui o texto...'
                )

                contexto = gr.Textbox(
                    label='3). Contexto (opcional)',
                    placeholder='📝 Insira aqui o contexto...'
                )

            chatbot = gr.Chatbot(
                label="Histórico do Chat",
                type="messages",
                avatar_images=(
                    '/ouvdia/assets/avatar_usuario.png',
                    '/ouvdia/assets/avatar_chatbot.svg'
                ),
            )

            # Estado inicial
            estado = gr.State({"client_ip": None, "usuario_logado": None})

            # Componentes para exibição
            ip_info = gr.Markdown()
            user_info = gr.Markdown()

            # Carrega e atualiza o estado na inicialização
            interface_chat.load(self.atualizar_estado, inputs=[estado], outputs=[ip_info, user_info])

            # Integração com função de feedback de likes no Chatbot (Like/Dislike)
            chatbot.like(
                fn=self.processar_feedback,
                inputs=[chatbot, estado],
                outputs=[]
            )

            # Integração da função Like/Dislike
            #chatbot.like(self.aplicar_gostei, None, None)


            # Integração do botão de desfazer
            chatbot.undo(self.aplicar_desfazer, chatbot, [chatbot, texto_principal])

            # (Opcional) Retirar ou ajustar o retry
            # chatbot.retry(self.aplicar_refazer, [chatbot, prompt_usuario, texto_principal, contexto], [chatbot])

            # Submit do 'texto_principal' -> chama _gerar() -> atualiza chatbot
            texto_principal.submit(
                self._gerar,
                [chatbot, prompt_usuario, texto_principal, contexto],
                [chatbot],
                queue=True
            )

            # Limpa o campo texto_principal depois de enviar
            texto_principal.submit(lambda: "", None, [texto_principal])

        # Retorna o Blocks (que é o front da aplicação Gradio)
        return interface_chat


# Execução do código
if __name__ == "__main__":

    # ========== Carregar variáveis de ambiente ==========
    try:
        # Carrega o arquivo HUGGINGFACE_TOKEN.env que contém variáveis de ambiente
        load_dotenv('/ouvdia/HUGGINGFACE_TOKEN.env')

        # Carrega o token do HuggingFace
        hf_token = os.getenv('HUGGINGFACE_TOKEN')

        # Verifica se o token foi carregado corretamente
        if not hf_token:
            raise ValueError("A variável de ambiente 'HUGGINGFACE_TOKEN' não foi definida. "
                             "Certifique-se de que ela está presente no arquivo HUGGINGFACE_TOKEN.env.")

        # Carrega o arquivo USUARIOS_VALIDOS.env que contém variáveis de ambiente
        load_dotenv('/ouvdia/USUARIOS_VALIDOS.env')

        # Carrega a variável contendo os usuários válidos
        usuarios_validos_raw = os.getenv('USUARIOS_VALIDOS')

        # Verifica se a variável foi carregada corretamente
        if not usuarios_validos_raw:
            raise ValueError("A variável de ambiente 'USUARIOS_VALIDOS' não foi definida ou está vazia. "
                             "Adicione-a ao arquivo .env no formato 'USUARIOS_VALIDOS=usuario1:senha1,usuario2:senha2'.")

        # Processa a string para transformá-la em uma lista de tuplas no formato (usuario, senha)
        usuarios_validos = []
        for u in usuarios_validos_raw.split(","):
            if ":" in u:  # Verifica se cada entrada tem o formato esperado
                usuarios_validos.append(tuple(u.split(":")))
            else:
                raise ValueError(f"A entrada '{u}' em 'USUARIOS_VALIDOS' está no formato incorreto. "
                                 "Cada entrada deve estar no formato 'usuario:senha'.")

        # Exibe os usuários válidos para conferência
        #print("Usuários válidos carregados com sucesso:")
        #for usuario, senha in usuarios_validos:
        #    print(f" - Usuário: {usuario}, Senha: {senha}")

    except Exception as e:
        # Exibe mensagens de erro detalhadas
        print(f"Erro ao carregar variáveis de ambiente: {e}")

    # ========== Inicialização do modelo e da interface ==========

    # Logando no HuggingFace Hub (caso necessário para acessar modelos privados)
    login(token=hf_token)

    # Inicializa o modelo LLM
    llm = ModeloLinguagemContínuo(
        modelo="gildocouto/Llama-3-8B-Instruct-AWQ-4bit",
        quantizacao="awq_marlin",
        tipo_dados="float16",
    )

    # Obtém o tokenizador do modelo
    tokenizador = llm.llm_motor.tokenizer.tokenizer

    # Define os parâmetros de amostragem
    parametros_amostragem = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
        stop_token_ids=[
            tokenizador.eos_token_id,
            tokenizador.convert_tokens_to_ids("<|eot_id|>")
        ],
    )
    
    # Inicializa a classe da interface
    interface_usuario = InterfaceUsuario(llm, tokenizador, parametros_amostragem)

    # Cria o Gradio Blocks (interface principal)
    interface_pronta = interface_usuario.iniciar()

    # ========== Lançar a interface com autenticação com tuplas de usuários válidos ==========
    imagem_icone_url = '/ouvdia/assets/rfb_logo_1.ico'
    interface_pronta.launch(
    share=False, 
    auth=usuarios_validos,
    auth_message="💡 Bem-vindo à OuvdIA – Assistente Inteligente da Ouvidoria da RFB!",
    favicon_path=imagem_icone_url,
    server_name="0.0.0.0", 
    server_port=7860, 
    show_error=True
    )