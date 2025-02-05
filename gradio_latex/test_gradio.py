import gradio as gr
from gradio import ChatMessage


def chatbot_response(user_message, history):
    assistant_reply = r"\[x^2 + x - 9 = 0\] \( a = 1 \)"
    assistant_reply= r"**Statement**: Every continuous function \( f \) on a closed interval \([a, b]\) is bounded and attains its bounds. That is, there exist points \( c, d \in [a, b] \) such that \( f(c) \leq f(x) \leq f(d) \) for all \( x \in [a, b] \)."
    return ChatMessage(content=assistant_reply)

delimiters = [
    {"left": r'\(', "right": r'\)', "display": True},
    {"left": r'\[', "right": r'\]', "display": True},
]

# Gradio Interface
with gr.Blocks(title="Chatbot") as iface:
    chatbot = gr.Chatbot(elem_id="chatbot", latex_delimiters=delimiters)

    gr.ChatInterface(fn=chatbot_response, type="messages", chatbot=chatbot)

# Launch the interface
iface.launch(share=True)
