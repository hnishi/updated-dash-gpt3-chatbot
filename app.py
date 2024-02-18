from textwrap import dedent

import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# client = OpenAI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


def textbox(text, box="AI", name="Philippe"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        thumbnail = html.Img(
            src=app.get_asset_url("Philippe.jpg"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-right": 5,
                "float": "left",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="light", inverse=False)

        return html.Div([thumbnail, textbox])

    else:
        raise ValueError("Incorrect option for `box`.")


description = """
Philippe is the principal architect at a condo-development firm in Paris. He lives with his girlfriend of five years in a 2-bedroom condo, with a small dog named Coco. Since the pandemic, his firm has seen a  significant drop in condo requests. As such, he’s been spending less time designing and more time on cooking,  his favorite hobby. He loves to cook international foods, venturing beyond French cuisine. But, he is eager  to get back to architecture and combine his hobby with his occupation. That’s why he’s looking to create a  new design for the kitchens in the company’s current inventory. Can you give him advice on how to do that?
"""

# Authentication

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Load images
IMAGES = {"Philippe": app.get_asset_url("Philippe.jpg")}


# Define Layout
conversation = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        # q: What does the following line do?
        # a: It sets the height of the div to 90% of the viewport height minus 132px
        # "height": "calc(90vh - 132px)",
        "height": "calc(80vh - 132px)",
        "flex-direction": "column-reverse",
    },
)

api_key_input = dbc.InputGroup(
    children=[
        dbc.Input(
            id="api-key-input", placeholder="Set your OpenAI API Key", type="text"
        ),
    ]
)

api_key_alert = dbc.Alert(
    "API Key is not set.",
    id="api-key-alert",
    dismissable=True,
    fade=False,
    is_open=True,
    color="danger",
)


controls = dbc.InputGroup(
    children=[
        dbc.Input(id="user-input", placeholder="Write to the chatbot...", type="text"),
        dbc.Button("Submit", id="submit"),
        # dbc.InputGroupAddon(dbc.Button("Submit", id="submit"), addon_type="append"),
    ]
)

app.layout = dbc.Container(
    fluid=False,
    children=[
        # q: How can I change the logo?
        # a: Replace the file dash-logo.png in the assets folder
        Header("Dash GPT-3 Chatbot", app),
        # q: what does the following line do?
        # a: it adds a horizontal line
        api_key_input,
        api_key_alert,
        html.Hr(),
        dcc.Store(id="store-conversation", data=""),
        conversation,
        controls,
        dbc.Spinner(html.Div(id="loading-component")),
    ],
)


@app.callback(
    Output("api-key-alert", "is_open"),
    [Input("api-key-input", "value")],
    [State("api-key-alert", "is_open")],
)
def toggle_alert_no_fade(api_key, is_open):
    if api_key is not None and api_key != "":
        return False
    return True


@app.callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="user") if i % 2 == 0 else textbox(x, box="AI")
        for i, x in enumerate(chat_history.split("<split>")[:-1])
    ]


@app.callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks, n_submit):
    return ""


@app.callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    # q: What does the following line do?
    # a: It triggers the callback when the button with id "submit" is clicked
    # q: Describe Input() function arguments
    # a: The first argument is the id of the component that triggers the callback.
    #   The second argument is the property of the component that triggers the callback.
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [
        State("user-input", "value"),
        State("store-conversation", "data"),
        State("api-key-input", "value"),
    ],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history, api_key):
    if api_key is None or api_key == "":
        return chat_history, None
    client = OpenAI(api_key=api_key)

    if n_clicks == 0 and n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None

    name = "Philippe"

    prompt = dedent(
        f"""
        {description}

        You: Hello {name}!
        {name}: Hello! Glad to be talking to you today.
        """
    )

    # First add the user input to the chat history
    chat_history += f"You: {user_input}<split>{name}:"

    model_input = prompt + chat_history.replace("<split>", "\n")
    # response = client.completions.create(
    #     engine="davinci",
    #     prompt=model_input,
    #     max_tokens=250,
    #     stop=["You:"],
    #     temperature=0.9,
    # )
    messages = [
        {
            "role": "user",
            "content": model_input,
        }
    ]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    print(response)
    model_output = response.choices[0].message.content

    chat_history += f"{model_output}<split>"

    return chat_history, None


if __name__ == "__main__":
    # app.run_server(host="0.0.0.0", port=8080, debug=False)
    app.run_server(port=8050, debug=True, host="0.0.0.0", use_reloader=True)
