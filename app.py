import dash
import dash_bootstrap_components as dbc
from dash import html, dcc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}], use_pages=True)
app.title = 'Duplicate Registry Entry Dashboard'
server = app.server

app.layout = html.Div([
    dbc.Navbar(
        dbc.Container([
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.Button(children=[html.Img(src='https://www.vumc.org/marketing-engagement/sites/default/files/VUMC_Logo.jpg',
                                                              height='50px')], href='/', outline=True))
                    ], align='center', no_gutters=True
                ),
            ),
            dbc.Container([dbc.NavbarBrand([html.H3("Registry Duplicate Tracker")])],
                          className='d-flex justify-content-center'),
            dbc.NavbarToggler(id="navbar-toggler2"),
        ], fluid=True), color="light", className="mb-5 justify-content-between"),
    dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)
