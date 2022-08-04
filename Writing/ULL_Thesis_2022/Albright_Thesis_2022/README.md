

## Welcome to the ULL thesis Template

_Forrest Montgomery Jul 17, 2017_

_Modified by Daniel Newman Apr 4, 2018_

_Modified by Gerald Eaglin Sept 10, 2018_

This latex template attempts to aid in the formatting of a ULL style thesis.

Some helpful latex tips:

Give up on getting personally stylistic with this thesis. The Grad School wants the document formatted according to the [guidelines](http://www.ucs.louisiana.edu/~dpd0909/4/Guidelines4.html#p=1). However, you only need to provide one copy of the correctly formatted thesis.

### Creating Figures

```tex
\begin{figure}[b!]
\centering
  \includegraphics[width=0.5\textwidth]{Figures/wreckage.jpg}
  \caption{Wreckage of the Silver Bridge \cite{wreckage}}
  \label{fig:wreckage}
\end{figure}
```

### Subfigures (adjust the 0.52 to make the figures fit)

```tex
\begin{figure}
\centering
\begin{subfigure}{0.52\textwidth}
\includegraphics[width=\linewidth]{Figures/low_pen.pdf}
\caption{Low Mode}
\label{fig:low_mode_pen}
\end{subfigure}

\begin{subfigure}{0.52\textwidth}
\includegraphics[width=\linewidth]{Figures/middle_pen.pdf}
\caption{Middle Mode}
\label{fig:middle_mode_pen}
\end{subfigure}

\begin{subfigure}{0.52\textwidth}
\includegraphics[width=\linewidth]{Figures/high_pen.pdf}
\caption{High Mode}
\label{fig:high_mode_pen}
\end{subfigure}

\caption{Modal Frequencies Varying Across Workspace}
\label{fig:nat_freq_pen}
\end{figure}
```

### Minipage for separate but related figures

```tex
\begin{figure}[tb]
\begin{center}
  \begin{minipage}{0.45\columnwidth}
  \begin{center}
  \includegraphics[width = \textwidth]{Figures/Chapter1_fig/Discrete_Grid_lines_2}
  \caption{Grid representation}
  \label{fig:Grid}
  \end{center}
  \end{minipage}
\hspace{0.07\textwidth}
  \begin{minipage}{0.45\textwidth}
  \begin{center}
  \includegraphics[width = \columnwidth]{Figures/Chapter1_fig/Polygon_workspace_2}
  \caption{Polygonal representation}
  \label{fig:Polygonal}
  \end{center}
  \end{minipage}
\end{center}
\vspace{-0.2in}
\end{figure}
```

### Multiline Math

```tex
\begin{equation}
\begin{multlined}[t]
\label{eq:q_coefficnets}
g_0^i U_0(s_i)(a_{00} + a_{10}e^{-s_iT}+...+a_{\ell0}e^{-s_i \ell T})\\
+g_1^i U_1(s_i)(a_{01} + a_{11}e^{-s_iT}+...+a_{\ell1}e^{-s_i \ell T}) + ...\\
g_m^i U_m(s_i)(a_{0m} + a_{1m}e^{-s_iT}+...+a_{\ell m}e^{-s_i \ell T}) = 0
\end{multlined}
\end{equation}
```
### Huge Multiline Math

_use the package breqn which will automatically break the equation_

```tex
\begin{dmath}
% really long equation
\end{dmath}
```


### Annotating Images

```tex
\begin{figure}[b!]
\begin{minipage}[b!]{1\textwidth}
\centering
\makebox[0pt]{
\begin{tikzpicture}
    \node[anchor=south west,inner sep=0] (image) at (0,0,0) {\includegraphics[width=4.5in]{Figures/Planar_Setup.JPG}};
    \begin{scope}[x={(image.south east)},y={(image.north west)}]
        %% next four lines will help you to locate the point needed by forming a grid. comment these four lines in the final picture:
       % \draw[help lines,xstep=.1,ystep=.1] (0,0) grid (1,1);
       % \draw[help lines,xstep=.05,ystep=.05] (0,0) grid (1,1);
       % \foreach \x in {0,1,...,9} { \node [anchor=north] at (\x/10,0) {0.\x}; }
       % \foreach \y in {0,1,...,9} { \node [anchor=east] at (0,\y/10) {0.\y};}
        %% upto here
        \draw[orange, dashed, ultra thick, text=black] (0.51, 0.3) -- +(2.3in,0)node[anchor=west] {Payload};
        \draw[orange, dashed, ultra thick, text=black] (0.27,0.7) -- +(-1.4in,0)node[anchor=east] {Cable};
        \draw[orange, dashed, ultra thick, text=black] (0.12,0.89) -- +(-0.8in,0)node[anchor=east] {Pulley};
        \draw[orange, dashed, ultra thick, text=black] (0.14,0.1) -- +(-0.7in,0in)node[anchor=east] {Motor Box};
        \draw[orange, dashed, ultra thick, text=black] (0.88,0.89) -- +(0.7in,0)node[anchor=west] {Pulley};
        \draw[orange, dashed, ultra thick, text=black] (0.75, 0.7) -- +(1.3in,0)node[anchor=west] {Cable};
        \draw[orange, dashed, ultra thick, text=black] (0.88, 0.1) -- +(0.7in,0)node[anchor=west] {Motor Box};
    \end{scope}
\end{tikzpicture}
}\par
\end{minipage}
\caption{Planar Cable Suspended Parallel Manipulator Experimental Setup}
\label{fig:planar_setup}
\end{figure}
```