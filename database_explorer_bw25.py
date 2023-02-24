# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:14:57 2022

@author: romai
"""
# Importing libraries
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
import matplotlib.backends.backend_pdf
import plotly.graph_objects as go
import plotly.express as px
import bw2analyzer as bwa
import bw2data as bwd
import bw2calc as bwc
import bw2io as bwio
from IPython.display import display

from dashboard_functions import calculate_DashBoard, plot_DashBoard

# Default impact categories
methods_EF = [
    m
    for m in bwd.methods
    if "EF v3.0 EN15804" in str(m)
    and not "no LT" in str(m)
    and not "obsolete" in str(m)
]
methods_CC = [m for m in methods_EF if "climate" in str(m)]
method_CC = methods_CC[0]


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def Contribution_Analysis_by_Substances(lca, ratio=0.8, length_max=10):
    """Function returning a dataframe giving the impact contribution by substances"""
    df_characterized_inventory = lca.to_dataframe(
        matrix_label="characterized_inventory"
    )
    df_CA_substance = df_characterized_inventory.groupby("row_name")[["amount"]].sum()
    df_CA_substance = df_CA_substance.sort_values(by="amount", ascending=False)

    impact = lca.score
    impact_i = 0
    i = 0
    while impact_i < ratio * impact:
        impact_i += df_CA_substance.iloc[i]["amount"]
        i += 1

    i = np.min([i, length_max])
    df_CA_substance = df_CA_substance.iloc[0:i]
    df_CA_substance.loc["Other"] = impact - df_CA_substance.amount.sum()
    df_CA_substance["percentage"] = df_CA_substance.amount / impact * 100

    return df_CA_substance


def Contribution_Analysis_by_Activities(lca, ratio=0.8, length_max=10):
    """Function returning a dataframe giving the impact contribution by activities"""
    df_characterized_inventory = lca.to_dataframe(
        matrix_label="characterized_inventory"
    )
    df_CA_activity = df_characterized_inventory.groupby("col_name")[["amount"]].sum()
    df_CA_activity = df_CA_activity.sort_values(by="amount", ascending=False)

    impact = lca.score
    impact_i = 0
    i = 0
    while impact_i < ratio * impact:
        impact_i += df_CA_activity.iloc[i]["amount"]
        i += 1

    i = np.min([i, length_max])
    df_CA_activity = df_CA_activity.iloc[0:i]
    df_CA_activity.loc["Other"] = impact - df_CA_activity.amount.sum()
    df_CA_activity["percentage"] = df_CA_activity.amount / impact * 100

    return df_CA_activity


class list_act:
    def __init__(self, database, name, location="", unit="", list_act_input=False):
        self.database = database
        self.name = name
        self.list_act = list_act_input
        self.location = location
        self.unit = unit

    def search(self, strict=False):
        if not self.list_act:
            list_act = [
                act
                for act in self.database
                if self.name in act["name"]
                and self.location in act["location"]
                and self.unit in act["unit"]
            ]
            if strict:
                list_act = [act for act in list_act if act["name"] == self.name]
            self.list_act = list_act

    def get_list(self, field):
        return set([act[field] for act in self.list_act])

    def get_lists(self):
        self.list_name = self.get_list(field="name")
        self.list_location = self.get_list(field="location")
        self.list_unit = self.get_list(field="unit")

    def print_lists(self):
        print(color.BOLD + color.UNDERLINE + "List of names:" + color.END)
        display(self.list_name)
        print(color.BOLD + color.UNDERLINE + "List of locations:" + color.END)
        display(self.list_location)
        print(color.BOLD + color.UNDERLINE + "List of units:" + color.END)
        display(self.list_unit)

    def get_comment(self, i):
        act = self.list_act[i]
        print(color.BOLD + color.UNDERLINE + str(act) + color.END)
        display(act["comment"])

    def get_comments(self):
        for i, act in enumerate(self.list_act):
            self.get_comment(i)
            print("\n")

    def get_inventory(self, i):
        df = (
            self.list_act[i]
            .exchanges()
            .to_dataframe()[
                [
                    "source_name",
                    "source_location",
                    "source_unit",
                    "edge_amount",
                    "edge_type",
                ]
            ]
        )

        df["edge_type"] = pd.Categorical(
            df["edge_type"], ["production", "technosphere", "biosphere"]
        )
        df = df.sort_values("edge_type")
        return df

    def get_inventories(self, index_name="name"):
        DF = pd.DataFrame()
        for i, act in enumerate(self.list_act):
            df = self.get_inventory(i)
            df = df.groupby("source_name")["edge_amount"].sum()
            DF[i] = df
        if index_name == "location":
            DF.columns = [act["location"] for act in self.list_act]
        if index_name == "name":
            DF.columns = [act["name"] + "_" + act["location"] for act in self.list_act]
        self.DF = DF

    def get_impacts(self, methods=methods_EF):
        list_inv = [{act: 1} for act in self.list_act]
        bwd.calculation_setups["multiLCA"] = {"inv": list_inv, "ia": methods}
        myMultiLCA = bwc.MultiLCA("multiLCA")
        df_impacts = pd.DataFrame(data=myMultiLCA.results)
        df_impacts.columns = [f"{m[1]} \n {m[2]}" for m in methods]
        df_impacts.index = [
            f"{act['name']} [{act['location']}]" for act in self.list_act
        ]

        self.impacts = df_impacts
        self.methods = methods
        return df_impacts

    def plot_impact_climate(self):
        df = self.impacts

        list_col = [col for col in df.columns if "climate change" in col]
        df = df[list_col]
        max_value = df.max().max()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax2 = ax.twinx()
        ax2.grid(False)
        df.T.plot(ax=ax, kind="bar", alpha=0.6, rot=15)
        ax2.set_zorder(-1)
        (df.T / max_value * 100).plot(ax=ax2, kind="bar", alpha=0.0, rot=15)
        ax2.set_ylabel("Percentage of maximum value (%)")
        if len(self.list_unit) == 1:
            ax.set_ylabel("kg$CO_2$eq/%s" % list(self.list_unit)[0])
        else:
            ax.set_ylabel("kg$CO_2$eq/%s" % str(self.list_unit))
        ax.set_title("Carbon footprint of %s" % self.name)
        return fig, ax

    def plot_impacts(self, double_axis=True):
        fig, ax = plt.subplots(5, 4, sharex=True, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.4, wspace=0.6)
        ax = ax.ravel()
        for axi in ax[self.impacts.shape[1] :]:
            axi.axis("off")
        ax = ax[0 : self.impacts.shape[1]]
        if double_axis:
            ax2 = [axi.twinx() for axi in ax]
        self.impacts.plot(
            ax=ax, legend=False, subplots=True, kind="bar", alpha=0.6,
        )
        if double_axis:
            (self.impacts / self.impacts.max() * 100).plot(
                ax=ax2, legend=False, subplots=True, kind="bar", alpha=0
            )
        for i, axi in enumerate(ax):
            if double_axis:
                ax2[i].grid(False)
                ax2[i].set_zorder(-1)
                ax2[i].set_ylabel("% of max value")
                ax2[i].set_title("")
            m = self.methods[i]
            # m = m[1]+'\n'+m[2]
            axi.set_title(m[1].replace(":", "\n"), fontsize=13)
            axi.set_ylabel(bwd.Method(self.methods[i]).metadata["unit"])

        return fig, ax

    def explore(self, strict=False, comments=False):
        self.search(strict=strict)
        if comments:
            display(self.list_act)
            print()
        self.get_lists()
        self.print_lists()
        if comments:
            self.get_comments()

    def analyse(self, methods_CC=methods_CC, methods_EF=methods_EF, print_data=True):
        self.get_inventories()
        if print_data:
            print(color.BOLD + color.UNDERLINE + "All flows:" + color.END)
            display(self.DF)
        self.get_impacts(methods=methods_CC)
        if print_data:
            print(color.BOLD + color.UNDERLINE + "Carbon footprint:" + color.END)
            display(self.impacts)
        self.get_impacts(methods_EF)
        if print_data:
            print(color.BOLD + color.UNDERLINE + "All impacts:" + color.END)
            display(self.impacts)
        fig, ax = self.plot_impact_climate()
        fig, ax = self.plot_impacts()

    def calculate_contribution_analysis(
        self, act, method, ratio=0.8, length_max=10, amount=1
    ):
        lca = bwc.LCA({act: amount}, method)
        lca.lci()
        lca.lcia()

        df_CAe = Contribution_Analysis_by_Substances(
            lca=lca, ratio=ratio, length_max=length_max
        )
        df_CAp = Contribution_Analysis_by_Activities(
            lca=lca, ratio=ratio, length_max=length_max
        )
        self.df_CAe = df_CAe
        self.df_CAp = df_CAp
        self.lca_score = lca.score

    def plot_contribution_analysis(self, act, method, amount):
        fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        self.df_CAe[["percentage"]].T.plot(
            ax=ax[0], kind="barh", stacked=True, alpha=0.6, rot=90
        )
        ax[0].set_title("Contribution per substances")
        ax[0].set_xlabel("Percentage (%)")
        ax[0].legend(bbox_to_anchor=(1.173, 1))

        self.df_CAp[["percentage"]].T.plot(
            ax=ax[1], kind="barh", stacked=True, alpha=0.6, rot=90
        )
        ax[1].legend(bbox_to_anchor=(1, 1))
        ax[1].set_title("Contribution per activities")
        ax[1].set_xlabel("Percentage (%)")

        m = bwd.Method(method)
        title = (
            'Contribution analysis for "%s"' % act["name"]
            + "\n for the impact category: %s" % str(method)
            + "\n Total impact = %0.2g %s/ %0.f %s"
            % (self.lca_score, m.metadata["unit"], amount, act["unit"])
        )
        plt.suptitle(title, fontsize=20, y=1.07)
        return fig

    def contribution_analysis(
        self, i, methods, ratio=0.8, length_max=10, amount=1, save=False
    ):
        act = self.list_act[i]
        print(act)
        if save:
            pdf = matplotlib.backends.backend_pdf.PdfPages(
                "output_contribution_analysis.pdf"
            )
        for method in methods:
            print(method)
            self.calculate_contribution_analysis(
                act=act,
                method=method,
                ratio=ratio,
                length_max=length_max,
                amount=amount,
            )
            fig = self.plot_contribution_analysis(act=act, method=method, amount=amount)
            if save:
                pdf.savefig(fig, bbox_inches="tight")
        if save:
            pdf.close()

    def DashBoard(self, i, method, cutoff, amount=1):
        act = self.list_act[i]
        print(act)
        print(method)

        lca = bwc.LCA({act: amount}, method)
        lca.lci()
        lca.lcia()

        print(lca.score)

        (
            df,
            lca,
            fig_sunburst_pos,
            fig_sunburst_neg,
            fig_waterfall,
            fig_sankey,
        ) = calculate_DashBoard(lca, cutoff)

        app = plot_DashBoard(
            df, lca, fig_sunburst_pos, fig_sunburst_neg, fig_waterfall, fig_sankey
        )
        return app

    def DashBoardFromJSON(self):
        (
            df,
            fig_sunburst_pos,
            fig_sunburst_neg,
            fig_waterfall,
            fig_sankey,
        ) = calculate_DashBoardFromJSON("kg CO2eq")

        app = plot_DashBoard(
            df, lca, fig_sunburst_pos, fig_sunburst_neg, fig_waterfall, fig_sankey
        )
        return app

    # Functions prior to the dashboard
    def plot_sankey(self, i, method, cutoff, amount=1):
        act = self.list_act[i]
        lca = bwc.LCA({act: amount}, method)
        lca.lci()
        lca.lcia()
        impact = lca.score
        unit = bwd.Method(method).metadata["unit"]

        gt = GraphTraversal().calculate({act: amount}, method=method, cutoff=cutoff)

        acts = gt["lca"].activity_dict
        id_to_key = {v: k for k, v in acts.items()}
        # labels = {k: bw.get_activity(v)["name"] for k in gt["nodes"].values()}
        ids = list(gt["nodes"].keys())
        labels = [bwd.get_activity(id_to_key[id])["name"] for id in ids[1:]]
        labels = ["root"] + labels
        id_to_idx = {id: idx for idx, id in enumerate(ids)}
        edges = gt["edges"]
        edges_plot = dict(
            target=[id_to_idx[edge["to"]] for edge in edges],
            source=[id_to_idx[edge["from"]] for edge in edges],
            value=[edge["impact"] / impact * 100 for edge in edges],
        )

        fig = go.Figure(
            data=[
                go.Sankey(
                    valueformat=".0f",
                    valuesuffix=" %",
                    node=dict(label=labels),
                    link=edges_plot,
                ),
            ],
        )

        if True:
            fig.update_layout(
                title_text="%s : \n%0.3f %s/%0.f %s"
                % (str(method), impact, unit, amount, act["unit"]),
                title_x=0.5,
                font_size=12,
                width=800,
                height=500,
            )

        return fig

    def plot_sankeys(self, i, methods, cutoff, amount=1, save=True):
        with open("output_Sankey.html", "a") as f:
            for method in methods:
                print(method)
                fig = self.plot_sankey(i=i, method=method, cutoff=cutoff, amount=amount)
                if save:
                    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
                else:
                    fig.show()

    def plot_sunburst(self, i, method, cutoff, amount=1):
        act = self.list_act[i]
        lca = bwc.LCA({act: amount}, method)
        lca.lci()
        lca.lcia()
        impact = lca.score
        unit = bw.Method(method).metadata["unit"]

        gt = GraphTraversal().calculate({act: amount}, method=method, cutoff=cutoff)

        # Label
        acts = gt["lca"].activity_dict
        id_to_key = {v: k for k, v in acts.items()}

        ids = list(gt["nodes"].keys())
        labels = [bw.get_activity(id_to_key[id])["name"] for id in ids[1:]]
        labels_loc = [bw.get_activity(id_to_key[id])["location"] for id in ids[1:]]
        labels = [
            label + " " + label_loc for label, label_loc in zip(labels, labels_loc)
        ]
        # Source
        edges = gt["edges"]
        edges = edges[: len(labels)]
        ids_source = [edge["to"] for edge in edges]
        ids_source = [
            bw.get_activity(id_to_key[id])["name"]
            + " "
            + bw.get_activity(id_to_key[id])["location"]
            for id in ids_source[1:]
        ]
        ids_source = [""] + ids_source
        # Value
        value = [edge["impact"] for edge in edges]
        # Data dictionary
        data = dict(label=labels, location=labels_loc, parent=ids_source, value=value)
        df = pd.DataFrame.from_dict(data)
        # Percentage data
        df["value_pct"] = df.value / impact * 100
        # Flooring percentage value to avoid sum of childs slightly higher than parents, and calculating value from floored pct
        df.value_pct = (df.value_pct * 10).apply(math.floor) / 10
        df.value = df.value_pct / 100 * impact
        df.value = df.value.apply(lambda x: round(x, 3))

        # Filtering negative value
        list_label_negatif = list(df[df.value <= 0].label.unique())

        def is_parent_or_child_negative(row):
            if (row.label in list_label_negatif) or (row.parent in list_label_negatif):
                is_negative = True
            else:
                is_negative = False
            return is_negative

        df["is_negative"] = df.apply(is_parent_or_child_negative, axis=1)
        df_p = df[df.is_negative == False]
        df_n = df[df.is_negative == True]

        data_p = df_p.to_dict()
        data_n = df_n.to_dict()

        # Sunburst for positive value
        fig = px.sunburst(
            data_p,
            names="label",
            parents="parent",
            values="value",
            branchvalues="total",
            color="value_pct",
            color_continuous_scale="algae",
            hover_data=["location"],
            # valueformat = '.0f',
        )
        fig.update_traces(textinfo="percent parent")
        # fig.update_layout(uniformtext = dict(minsize=8, mode='hide'))

        title = f"{act['name'].capitalize()}: {impact:.2g} {unit}/{amount:2g}{act['unit']}\n <br> Method: {str(method)}"
        fig.update_layout(autosize=True, title_text=title, title_x=0.5, font_size=10)
        # fig.show()
        return fig

    def plot_sunbursts(self, i, methods, cutoff, amount=1, save=True):
        with open("output_Sunburst.html", "a") as f:
            for method in methods:
                print(method)
                fig = self.plot_sunburst(
                    i=i, method=method, cutoff=cutoff, amount=amount
                )
                if save:
                    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
                else:
                    fig.show()
