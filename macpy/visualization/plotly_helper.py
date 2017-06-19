import plotly
import plotly.plotly as py
from plotly.graph_objs import Layout, Font, Margin, XAxis, YAxis, Scatter, Figure
# https://plot.ly/python/user-guide/

#py.sign_in('Axioma01', 'rbgwcvrp8i')


def create_basic_layout(x_axis_title,
                        y_axis_title,
                        width=700,
                        height=600,
                        title=None,
                        x_axis_min=None,
                        x_axis_max=None,
                        y_axis_min=None,
                        y_axis_max=None,
                        font_size=18,
                        top=80,
                        bottom=80,
                        show_legend=True,
                        legend=None):
    """
    example usage:
    import macpy.visualization.plotly_helper as pl
    layout = pl.create_basic_layout('x axis title', 'y axis title')
    data = [pl.create_scatter_trace([1,2],[1,2]), pl.create_scatter_trace([1,2],[3,4])]
    figure = pl.create_figure(data, layout)
    pl.create_plot_offline(figure)
    """
    layout = Layout(title=title,
                    width=width,
                    height=height,
                    legend=dict(font=Font(family='Arial',
                                          size=font_size-1),
                                x=0,
                                y=-.45,
                                orientation='h'),
                    titlefont=Font(family='Arial',
                                   size=font_size,
                                   color='black'),
                    margin=Margin(l=80,
                                  r=80,
                                  b=bottom,
                                  t=top,
                                  pad=0),
                    xaxis=dict(title=x_axis_title,
                                titlefont=Font(family='Arial',
                                               size=font_size,
                                               color='black'),
                                tickfont=Font(family='Arial',
                                              size=font_size,
                                              color='black'),
                                range=[x_axis_min, x_axis_max] if x_axis_min is not None else None),
                    yaxis=YAxis(title=y_axis_title,
                                titlefont=Font(family='Arial',
                                               size=font_size,
                                               color='black'),
                                tickfont=Font(family='Arial',
                                              size=font_size,
                                              color='black'),
                                range=[y_axis_min, y_axis_max] if y_axis_min is not None else None),
                    showlegend=show_legend)

    return dict(layout)


def save_figure_as_png(filename, figure):
    py.image.save_as(figure, filename, format='png')


def create_scatter_trace(x_values, y_values, mode='markers', name=None):
    return Scatter(x=x_values, y=y_values, mode=mode, name=name)


def create_figure(data, layout):
    return Figure(data=data, layout=layout)


def create_plot_offline(figure):
    plotly.offline.iplot(figure)
