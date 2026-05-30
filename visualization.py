import plotly.express as px

def create_bar_chart(df, x, y):
    fig = px.bar(df, x=x, y=y, title=f"{y} by {x}", template="plotly_white")
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig

def create_line_chart(df, x, y):
    fig = px.line(df, x=x, y=y, title=f"{y} Trend Over {x}", template="plotly_white")
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig

def create_scatter_plot(df, x, y):
    fig = px.scatter(df, x=x, y=y, title=f"Relationship between {x} and {y}", template="plotly_white")
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig