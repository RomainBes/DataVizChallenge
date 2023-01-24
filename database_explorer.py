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
import brightway2 as bw
import bw2analyzer as bwa
from bw2calc import GraphTraversal
import lca_algebraic as agb
from IPython.display import display

# Default impact categories
methods_EF = [
    m
    for m in bw.methods
    if "EF v3.0 EN15804" in str(m)
    and not "no LT" in str(m)
    and not "obsolete" in str(m)
]
methods_CC = [m for m in methods_EF if "climate" in str(m)]


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


def CA_Elem_Flow(lca, ratio=0.8, length_max=10):

    # EF analysis
    CAe = bwa.ContributionAnalysis().annotated_top_emissions(lca)
    df_CAe = pd.DataFrame(CAe, columns=["lca_score", "inventory_amount", "activity"])
    index = [act[2]["name"] for act in CAe]
    df_CAe["activity"] = index

    df_CAe = df_CAe.groupby("activity").sum()
    df_CAe = df_CAe.sort_values(by="lca_score", ascending=False)

    impact = lca.score
    impact_i = 0
    i = 0
    while impact_i < ratio * impact:
        impact_i += df_CAe.iloc[i]["lca_score"]
        i += 1

    i = np.min([i, length_max])
    df_CAe = df_CAe.iloc[0:i]
    # display(df_CAe)
    df_CAe.loc["Other"] = [impact - df_CAe.lca_score.sum(), None]
    df_CAe["percentage"] = df_CAe.lca_score / impact * 100
    # df_CAe.index = df_CAe.activity

    return df_CAe


def CA_Process(lca, ratio=0.8, length_max=10):

    # Process analysis
    CAp = bwa.ContributionAnalysis().annotated_top_processes(lca)
    df_CAp = pd.DataFrame(CAp, columns=["lca_score", "inventory_amount", "activity"])
    index = [act[2]["name"] for act in CAp]
    df_CAp["activity"] = index

    df_CAp = df_CAp.groupby("activity").sum()
    df_CAp = df_CAp.sort_values(by="lca_score", ascending=False)

    impact = lca.score
    impact_i = 0
    i = 0
    while impact_i < ratio * impact:
        impact_i += df_CAp.iloc[i]["lca_score"]
        i += 1
        if i >= np.min([length_max, df_CAp.shape[0]]):
            break

    i = np.min([i, length_max])
    df_CAp = df_CAp.iloc[0:i]
    # display(df_CAe)
    df_CAp.loc["Rest"] = [impact - df_CAp.lca_score.sum(), None]
    df_CAp["percentage"] = df_CAp.lca_score / impact * 100
    # df_CAe.index = df_CAe.activity

    return df_CAp


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
        agb.printAct(self.list_act[i])

    def get_inventories(self):
        agb.printAct(*self.list_act)

    def get_inventory2(self, i):
        df = pd.DataFrame(index=["amount", "unit", "key", "type"])
        for exc in self.list_act[i].exchanges():
            exc_name = bw.get_activity(exc["input"])["name"]
            if exc_name in df.columns:
                df[exc_name].loc["amount"] += exc.amount
            else:
                df[exc_name] = [
                    np.float(exc.amount),
                    exc.unit,
                    exc.input.key,
                    exc["type"],
                ]
        return df.T

    def get_inventories2(self, index_name="name"):
        # List of exchanges
        DF = pd.DataFrame()
        DFt = pd.DataFrame()
        DFb = pd.DataFrame()

        for act in self.list_act:
            df = pd.DataFrame(index=["amount", "unit", "key", "type"])
            for exc in act.exchanges():
                exc_name = bw.get_activity(exc["input"])["name"]
                if exc_name in df.columns:
                    df[exc_name].loc["amount"] += exc.amount
                else:
                    df[exc_name] = [
                        np.float(exc.amount),
                        exc.unit,
                        exc.input.key,
                        exc["type"],
                    ]

            df = df.T[["amount", "type"]]
            DF = pd.concat([DF, df.amount], axis=1, sort=False)
            DFt = pd.concat(
                [DFt, df.amount[df.type == "technosphere"]], axis=1, sort=False
            )
            DFb = pd.concat(
                [DFb, df.amount[df.type == "biosphere"]], axis=1, sort=False
            )
        if index_name == "location":
            DF.columns = [act["location"] for act in self.list_act]
        if index_name == "name":
            DF.columns = [act["name"] + "_" + act["location"] for act in self.list_act]
        DFt.columns = DF.columns
        DFb.columns = DF.columns
        self.DF = DF
        self.DFt = DFt
        self.DFb = DFb
        return DF

    def get_impact(self, i, method):
        df = agb.multiLCAAlgebric(self.list_act[i], method)
        return df

    def get_impacts(self, methods=methods_EF):
        df = agb.multiLCAAlgebric(self.list_act, methods)
        df.columns = [m[1] + "\n" + m[2] for m in methods]
        self.impacts = df
        self.methods = methods
        return df

    def plot_impact_climate(self):
        df = self.impacts

        list_col = [col for col in df.columns if "climate change" in col]
        df = df[list_col]
        max_value = df.max().max()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax2 = ax.twinx()
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
                ax2[i].set_zorder(-1)
                ax2[i].set_ylabel("% of max value")
                ax2[i].set_title("")
            m = self.methods[i]
            # m = m[1]+'\n'+m[2]
            axi.set_title(m[1].replace(":", "\n"), fontsize=13)
            axi.set_ylabel(bw.Method(self.methods[i]).metadata["unit"])

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

    def analyse(
        self, methods_CC=methods_CC, methods_EF=methods_EF, strict=False, print_mode=3
    ):
        self.get_inventories()
        self.get_inventories2()
        if print_mode >= 1:
            print(color.BOLD + color.UNDERLINE + "All flows:" + color.END)
            display(self.DF)
        if print_mode >= 3:
            print(color.BOLD + color.UNDERLINE + "Technosphere flows:" + color.END)
            display(self.DFt)
            print(color.BOLD + color.UNDERLINE + "Biosphere flows:" + color.END)
            display(self.DFb)
        self.get_impacts(methods=methods_CC)
        if print_mode >= 2:
            print(color.BOLD + color.UNDERLINE + "Carbon footprint:" + color.END)
            display(self.impacts)
        self.get_impacts(methods_EF)
        if print_mode >= 4:
            print(color.BOLD + color.UNDERLINE + "All impacts:" + color.END)
            display(self.impacts)
        fig, ax = self.plot_impact_climate()
        fig, ax = self.plot_impacts()

    def calculate_contribution_analysis(
        self, act, method, ratio=0.8, length_max=10, amount=1
    ):
        lca = bw.LCA({act: amount}, method)
        lca.lci()
        lca.lcia()

        df_CAe = CA_Elem_Flow(lca=lca, ratio=ratio, length_max=length_max)
        df_CAp = CA_Process(lca=lca, ratio=ratio, length_max=length_max)
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

        m = bw.Method(method)
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

    def plot_sankey(self, i, method, cutoff, amount=1):
        act = self.list_act[i]
        lca = bw.LCA({act: amount}, method)
        lca.lci()
        lca.lcia()
        impact = lca.score
        unit = bw.Method(method).metadata["unit"]

        gt = GraphTraversal().calculate({act: amount}, method=method, cutoff=cutoff)

        acts = gt["lca"].activity_dict
        id_to_key = {v: k for k, v in acts.items()}
        # labels = {k: bw.get_activity(v)["name"] for k in gt["nodes"].values()}
        ids = list(gt["nodes"].keys())
        labels = [bw.get_activity(id_to_key[id])["name"] for id in ids[1:]]
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
        lca = bw.LCA({act: amount}, method)
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
        #Value
        value = [edge["impact"] for edge in edges]
        #Data dictionary
        data = dict(label = labels,
                    location = labels_loc,
                    parent = ids_source,
                    value = value)
        df = pd.DataFrame.from_dict(data)
        #Percentage data
        df['value_pct'] = df.value/impact * 100
        #Flooring percentage value to avoid sum of childs slightly higher than parents, and calculating value from floored pct
        df.value_pct = (df.value_pct*10).apply(math.floor)/10
        df.value = df.value_pct/100 * impact
        df.value = df.value.apply(lambda x:round(x,3))

        #Filtering negative value
        list_label_negatif = list(df[df.value <=0].label.unique())

        def is_parent_or_child_negative(row):
            if (row.label in list_label_negatif) or (row.parent in list_label_negatif):
                is_negative = True
            else:
                is_negative = False
            return is_negative

        df['is_negative'] = df.apply(is_parent_or_child_negative, axis = 1)
        df_p = df[df.is_negative == False]
        df_n = df[df.is_negative == True]

        data_p = df_p.to_dict()
        data_n = df_n.to_dict()

        #Sunburst for positive value
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
