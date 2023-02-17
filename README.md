# Contribution Analysis

This is a demo of a [`plotly`](https://dash.plotly.com/) dash visualisation based on [`brightway2`](https://brightway.dev) LCA calculations.

It was written for brightway2.5 and uses a modifiied version of the "AssumedDiagonalGraphTraversal" to obtain contributions of nodes and edges, and then splits those contribution into upstream paths to be able to create sunburst diagrams.


<div align="center">
<img src="https://github.com/RomainBes/DataVizChallenge/raw/main/assets/sample.png" height="300"/>
</div>


The current implementation is still a naive "hack" to show that it is working.


Recently we discovered the great [`polyviz`](https://github.com/romainsacchi/polyviz), which has several similar functionalities and might provide a better implementation of the recursive supply chain calculations.