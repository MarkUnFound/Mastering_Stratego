% etd.cls 8-December-1996
%
% CVS attributes:
%	$Id: etd.cls,v 1.6 2005/12/09 22:13:00 asioson Exp $
%	$Source: /home/barista/papers/sioson_dis/etd.cls,v $
%
% Class for Virginia Polytechnic Institute and State University theses
% (graduate and undergraduate), based on "Graduate Policies and
% Procedures Manual".

% Created by Scott A. Guyer as an update to 'vpithesis.sty'.  The 
% vpithesis style has become outdated and incompatible with LaTeX2e.
% The name change (to etd) is since this work is intended for use
% with the Virginia Tech Electronic Thesis and Dissertation (ETD)
% project.  So while the style file is getting a facelift to a 
% class file, it is also being modified to facilitate translation
% between LaTeX and SGML.  This is the goal.

% This file has a bit of a history.  The version with which I am 
% working was last updated by Farooq Azam of the VT EE department.
% Farooq modified Robert W. Hendricks' (of the VT Materials Engineering
% Department) original vpithesis.sty to work with LaTeX2e (the
% original version only worked with LaTeX 2.09). Robert's creation was
% based on the MIT style (prepared by Stephen Guildea) as well as the
% Stanford style (prepared by Howard Trickey). 

% Modifications by Craig A. Struble to make the title page ETD
% compliant.

% keywords added by Allan A. Sioson (9/28/2003) 
% updated to handle BSIT/BSCS/BSIS documentation formats (Allan Sioson)
% 02/10/2014 : updated to handle group SP (Allan Sioson)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{etd}[1996/12/08 VT ETD Class]
\LoadClass{report}

\RequirePackage{ifthen}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We want to use the 11pt fonts, so pass this
% to the report class.  Pass anything else to
% report class as well.

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\ExecuteOptions{11pt}
\ProcessOptions


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    MAIN CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%% Here is the code for the old vpi11.sty file.
%% \input vpi11.sty\relax
%% Don't know how much of this is already defined in the report
%% class. I'll add it as necessary.
%%-------------------------------------------------------------

%%--------------------------------------------------------------
% Define Virginia Tech abbreviations
\def\VPI{\rm Virginia Polytechnic Institute and State University}
\def\vpi{\rm VPI \& SU}

% Define Virginia Tech abbreviations
\def\ADNU{\rm Ateneo de Naga University}
\def\adnu{\rm ADNU}

% Define page layout

% Margins should be an inch and a half on the left margin, one inch
% on the right, and an inch and a quarter on the top and bottom.

\if@twoside
  \setlength{\oddsidemargin}{0.5in}
  \setlength{\evensidemargin}{0pt}
\else
  \setlength{\oddsidemargin}{0.5in}
  \setlength{\evensidemargin}{0.5in}
\fi
\setlength{\topmargin}{.25in}
\setlength{\headheight}{.125in}
\setlength{\headsep}{.20in}
\setlength{\footskip}{0.415in}  % ~30pts
\setlength{\textwidth}{6in}
\setlength{\textheight}{7.75in} % \textheight = 11.00 - (\footskip + \headsep
                                %  + \headheight + \topmargin) - 1.25 - 1.00
\setlength{\marginparwidth}{83pt}
\setlength{\marginparsep}{10pt}


% The main body of the text will be space and a half.  The abstract,
% vita, footnotes, and bibliography sections will be single spaced.

%%%% BEGIN doublespace.sty
% The following is the DOUBLESPACE.STY by Stephen Page  --
% (sdpage@uk.ac.oxford.prg) and contains pieces of the Stanford style.
% This style option produces a space and a half document by redefining
% the LaTeX parameter \baselinestretch to 1.5
%
% It also fixes a number of problems with standard LaTeX, as follows:
%
% 1. A new environment "singlespace" is provided, within which single
%    spacing will apply. (Note that on entering this environment the
%    \normalsize command is issued automatically.)
% 2. Space and a half is turned off within footnotes and floats (figures
%    and tables).
% 3. Proper double spacing happens below tabular environments and in other
%    places where LaTeX uses a strut.
% 4. Slightly more space is inserted before footnotes.
%
% Note: This version sets a baselinestretch of 1.5. If this is not
%    right, put  \setstretch{1.7}  (or some appropriate value) before the
%    \begin{document} command. To print the document with single spacing
%    again, without removing "singlespace" environments, just put a
%    \setstretch{1} at the top of the document.
%
% Set up spacing

\def\baselinestretch{1.5}

% New command "\setstretch" (slightly more mnemonic... and easier to type)

\def\setstretch#1{\renewcommand{\baselinestretch}{#1}}

% New environment "singlespace".
% The baselinestretch only takes effect at a size change, so do one.

\def\singlespace{\def\baselinestretch{1}\@normalsize}
\def\endsinglespace{}

% Stretch the baseline BEFORE calculating the strut size. This improves
% spacing below tabular environments etc.

%\def\@setsize#1#2#3#4{\@nomath#1\let\@currsize#1\baselineskip
%   #2\baselineskip\baselinestretch\baselineskip
%          \setbox\strutbox\hbox{\vrule height.7\baselineskip
%      depth.3\baselineskip width\z@}
%   \normalbaselineskip\baselineskip#3#4}

% Increase the space between last line of text and footnote rule.

\skip\footins 20pt plus4pt minus4pt

% Reset baselinestretch within footnotes and floats. Originally stolen
% from Stanford thesis style.

\def\@xfloat#1[#2]{\ifhmode \@bsphack\@floatpenalty -\@Mii\else
   \@floatpenalty-\@Miii\fi\def\@captype{#1}\ifinner
      \@parmoderr\@floatpenalty\z@
    \else\@next\@currbox\@freelist{\@tempcnta\csname ftype@#1\endcsname
       \multiply\@tempcnta\@xxxii\advance\@tempcnta\sixt@@n
       \@tfor \@tempa :=#2\do
                        {\if\@tempa h\advance\@tempcnta \@ne\fi
                         \if\@tempa t\advance\@tempcnta \tw@\fi
                         \if\@tempa b\advance\@tempcnta 4\relax\fi
                         \if\@tempa p\advance\@tempcnta 8\relax\fi
         }\global\count\@currbox\@tempcnta}\@fltovf\fi
    \global\setbox\@currbox\vbox\bgroup
    \def\baselinestretch{1}\@normalsize
    \boxmaxdepth\z@
    \hsize\columnwidth \@parboxrestore}
\long\def\@footnotetext#1{\insert\footins{\def\baselinestretch{1}\footnotesize
    \interlinepenalty\interfootnotelinepenalty
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
   \edef\@currentlabel{\csname p@footnote\endcsname\@thefnmark}\@makefntext
    {\rule{\z@}{\footnotesep}\ignorespaces
      #1\strut}}}

%%%% End doublespace.sty


% Define Figure  environment

% Figures and Tables appear on separate pages with figure numbers and
% captions at the bottom and table numbers and captions at the top. A
% box is drawn around the figure or table. If figure or table is small
% enough, we will allow two to a page. Use

% \fig{x}{y}{long_caption}{label}{short_caption}

% where x is the x-axis length of the figure
%       y is the y-axis length of the figure
%       long_caption is the caption which will appear with the figure
%       label is the label created for referencing the figure in the text
%       short_caption is the caption which will appear in the list of figures

\newcommand{\fig}[5]{
  \begin{figure}[btp]\centering
  \framebox[#1in]{\rule{0in}{#2in}}\caption[#5]{#3}\label{#4}
  \end{figure}
}

% End of formatting parameters %%

%-----------------------
%  Title Page commands
%-----------------------
% \title and \author already exist

% Define the field of the degree to be awarded and author's department
% It is assumed that the author's department is the same as the chairman's
\def\field#1{\gdef\@field{#1}}
\def\department#1{\gdef\@department{#1}}
\def\secnddept#1{\gdef\@secnddept{#1}}
\def\@secnddept{\@department}

% Month and year of degree
\def\degreemonth#1{\gdef\@degreemonth{#1}}
\def\degreeyear#1{\gdef\@degreeyear{#1}}

% Keywords
\def\keywords#1{\gdef\@keywords{Keywords: #1}}

% Define possible degrees to be awarded
\def\PhDdegree{\gdef\@degreeword{Dissertation\ } 
%  \gdef\@faculty{\VPI}
  \gdef\@faculty{\ADNU}
  \gdef\@degree{Doctor of Philosophy}
  \gdef\@copyrightword{dissertation\ }}
\def\MSdegree{\gdef\@degreeword{Thesis\ } 
%  \gdef\@faculty{\VPI}
  \gdef\@faculty{\ADNU}
  \gdef\@degree{Master of Science}
  \gdef\@copyrightword{thesis\ }}
\def\Mdegree{\gdef\@degreeword{Thesis\ } 
%  \gdef\@faculty{\VPI}
  \gdef\@faculty{\ADNU}
  \gdef\@degree{Master of}
  \gdef\@copyrightword{thesis\ }}
\def\BSthesis{\gdef\@degreeword{Senior Thesis\ } 
  \gdef\@faculty{\@department}
  \gdef\@degree{Bachelor of Science}
  \gdef\@copyrightword{report\ }}
\def\BSproject{\gdef\@degreeword{Capstone Project\ } 
  \gdef\@faculty{\@department}
  \gdef\@degree{Bachelor of Science}
  \gdef\@copyrightword{report\ }}
\def\BShonors{\gdef\@degreeword{Senior honors thesis\ }
  \gdef\@faculty{\@department}
  \gdef\@degree{Bachelor of Science (HONORS)}
  \gdef\@copyrightword{thesis\ }}

\def\@keywords{Keywords: *** Please add keywords (e.g. $\backslash${\tt keywords\{word1, word2, ...\}}) ***}
\def\@degreeword{???\ } \def\@faculty{*** Please define degree sought ***}
\def\@degree{???} \def\@copyrightword{???\ }

% Define Committee

%\def\chairman#1{\gdef\@chairman{\vspace{0.45in}
%\parbox{3.in}{\rule{3.in}{0.01in}\\
% \mbox{}\hfill {\small #1, Chairman}\hfill\mbox{}}}
% \gdef\@abstractchairman{Committee Chairman:\ #1\\ \@department}}

\def\chairman#1{\gdef\@chairman{\vspace{0.45in}
\parbox{3.in}{
 \mbox{}\hfill {#1, Chairman}\hfill\mbox{}}}
 \gdef\@abstractchairman{Committee Chairman:\ #1\\ Department of \@department}}

\def\chairmen#1#2{
 \gdef\@chairman{{#1, Co-chairman}\\
                 {#2, Co-chairman}}
 \gdef\@abstractchairman{Committee Co-chairmen:\\#1\\ \@department\medskip\\
  and\medskip\\#2\\ \@secnddept}}

\def\advisor#1{
 \gdef\@chairman{\vspace{0.45in}
 \parbox{3.in}{\rule{3.in}{0.01in}\\\\
 \mbox{}\hfill {Project Advisor:\ #1}\hfill\mbox{}\medskip}}
 \gdef\@abstractchairman{Project Advisor:\ #1\\ Department of \@department}
 \gdef\@projadvisor{#1}
 }

\def\advisors#1#2{
 \gdef\@chairman{\vspace{0.45in} \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small Co-advisor:\ #1} \hfill\mbox{}} \hfill
 \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small Co-advisor:\ #2} \hfill\mbox{}} \hfill}
 \gdef\@abstractchairman{Project Co-advisors:\\#1\\ \@department\medskip\\
  and\medskip\\#2\\ \@secnddept}}

\def\@chairman{*** You must define your committee chairman or advisor *** }
\def\@abstractchairman{*** You must define your committee chairman or advisor
 *** }

\def\studentsheader#1{
 \gdef\@studentsheader{#1}
}

\def\deanandchair#1#2{
 \gdef\@dean{#1}
 \gdef\@deptchair{#2}
}

%\def\@dean{Allan A. Sioson, PhD}
%\def\@deptchair{Joshua C. Martinez}

\def\member#1{\gdef\@member{\vspace{0.45in} \parbox{3.in}{\rule{3.in}{0.01in}\\
 \mbox{}\hfill {\small #1} \hfill\mbox{}}}}

\def\twomembers#1#2{
 \gdef\@member{\vspace{0.45in} \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #1} \hfill\mbox{}} \hfill
 \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #2} \hfill\mbox{}} \hfill}}

\def\threemembers#1#2#3{ 
 \gdef\@member{{#1}\\{#2}\\{#3}}
 \gdef\@firstmember{#1}
 \gdef\@secondmember{#2}
 \gdef\@thirdmember{#3} 
}

\def\onestudent#1{ 
 \gdef\@students{{#1}}
 \gdef\@firststudent{#1}
}

\def\twostudents#1#2{ 
 \gdef\@students{{#1} and {#2}}
 \gdef\@firststudent{#1}
 \gdef\@secondstudent{#2}
}

\def\threestudents#1#2#3{ 
 \gdef\@students{{#1}, {#2}, and {#3}}
 \gdef\@firststudent{#1}
 \gdef\@secondstudent{#2}
 \gdef\@thirdstudent{#3} 
}

\def\fourstudents#1#2#3#4{ 
 \gdef\@students{{#1}, {#2}, {#3}, and {#4}}
 \gdef\@firststudent{#1}
 \gdef\@secondstudent{#2}
 \gdef\@thirdstudent{#3} 
 \gdef\@fourthstudent{#4}
}

\def\onedegree#1{ 
 \gdef\@studentdegs{{#1}}
 \gdef\@firststudentdeg{#1}
}

\def\twodegrees#1#2{ 
 \gdef\@studentdegs{{#1}\\{#2}}
 \gdef\@firststudentdeg{#1}
 \gdef\@secondstudentdeg{#2}
}

\def\threedegrees#1#2#3{ 
 \gdef\@studentdegs{{#1}\\{#2}\\{#3}}
 \gdef\@firststudentdeg{#1}
 \gdef\@secondstudentdeg{#2}
 \gdef\@thirdstudentdeg{#3} 
}

\def\fourdegrees#1#2#3#4{ 
 \gdef\@studentdegs{{#1}\\{#2}\\{#3}\\{#4}}
 \gdef\@firststudentdeg{#1}
 \gdef\@secondstudentdeg{#2}
 \gdef\@thirdstudentdeg{#3} 
 \gdef\@fourthstudentdeg{#4}
}

%\def\fourmembers#1#2#3#4{
% \gdef\@member{\vspace{0.45in} \parbox{2.75in}{\rule{2.75in}{0.01in}\\
% \mbox{}\hfill {\small #1} \hfill\mbox{}\vspace{0.20in} \rule{2.75in}{0.01in}\\
% \mbox{}\hfill {\small #3} \hfill\mbox{}} \hfill
% \parbox{2.75in}{\rule{2.75in}{0.01in}\\
% \mbox{}\hfill {\small #2} \hfill\mbox{}\vspace{0.20in} \rule{2.75in}{0.01in}\\
% \mbox{}\hfill {\small #4} \hfill\mbox{}}}}

\def\fourmembers#1#2#3#4{ 
 \gdef\@member{ {#1}\\
 {#2}\\
 {#3}\\
 {#4}}
}

\def\fivemembers#1#2#3#4#5{
 \gdef\@member{\vspace{0.45in} \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #1} \hfill\mbox{}\vspace{0.20in} \rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #3} \hfill\mbox{}} \hfill
 \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #2} \hfill\mbox{}\vspace{0.20in} \rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #4} \hfill\mbox{}}\vspace{0.45in}
 \parbox{2.75in}{\rule{2.75in}{0.01in}\\
 \mbox{}\hfill {\small #5}\hfill\mbox{}\\}}}

% \def\@member{\em} % Not understood what this does 12/12/91

% Make Copyright Notice
% If there is no \copyright command, it is assumed that the student and
% VPI hold the copyright jointly. If you require modification of either
% of these statements, please contact the Virginia Tech Intellectual Properties
% office for advice on acceptable wording.

\def\permission{\par\noindent{\centering The author hereby grants to \ADNU\
permission to reproduce and \\* to distribute copies of this \@copyrightword
in whole or in part.}\par}
\def\copyrightnotice{\gdef\@copyrightnotice{\copyright\ \@degreeyear, \@students\par\permission}}
\def\@copyrightnotice{\small Copyright\ \@degreeyear, \@students}

% Make Titlepage
\def\maketitle{
\thispagestyle{empty}
\pagenumbering{roman}
\null
\begin{center}
\vspace*{-0.875in}\vfill
{\LARGE\bf \@title\par} \vfill
{\Large \@firststudent}\\
{\@degree~in \@firststudentdeg}\par\vfill
{\Large \@secondstudent}\\
{\@degree~in \@secondstudentdeg}\par\vfill
%{\Large \@thirdstudent}\\
%{\@degree~in \@thirdstudentdeg}\par\vfill
%{\Large \@fourthstudent}\\
%{\@degree~in \@fourthstudentdeg}\par\vfill
\@degreeword submitted to the faculty of the\\
Department of \@faculty\\
College of Computer Studies, \ADNU\\
in partial fulfillment of the requirements for their respective \\
Bachelor of Science degrees \vfill
%{\@degree}\smallskip\\%\smallskip\\
%\@field \vfill
\@chairman\\
\@member\par\vfill
\@degreemonth \space \@degreeyear\par
Naga City, Philippines\par\vfill
\@keywords\par\par
\bigskip
\@copyrightnotice\par
\end{center}
}


\def\makerecomm{
\newpage\thispagestyle{empty}
The Senior Project entitled
  {\large\bf\begin{center}\@title\end{center}}
\noindent 
developed by
  {\begin{center}{\large\bf\@firststudent}\\\@degree~in \@firststudentdeg\end{center}}
  {\begin{center}{\large\bf\@secondstudent}\\\@degree~in \@secondstudentdeg\end{center}}
  %{\begin{center}{\large\bf\@thirdstudent}\\\@degree~in \@thirdstudentdeg\end{center}}
  %{\begin{center}{\large\bf\@fourthstudent}\\\@degree~in \@fourthstudentdeg\end{center}}
\noindent
and submitted in partial fulfillment of the requirements
of their respective \@degree\ degrees
has been rigorously examined and recommended for approval and
acceptance.

\begin{center}
\begin{tabular}{ccc}
\\\\\\
\textbf{\@firstmember} 
 & ~~~~~~~~~~~~~~~~~~ & \textbf{\@secondmember}\\
Panel Member
 & & Panel Member\\
Date signed: \rule{3cm}{0.15mm}
 & & Date signed: \rule{3cm}{0.15mm}
\\\\\\
\textbf{\@thirdmember}
 &  ~~~~~~~~~~~~~~~~~~ & \textbf{\@projadvisor}\\
Panel Member
 & & Project Advisor\\
Date signed: \rule{3cm}{0.15mm}
 & & Date signed: \rule{3cm}{0.15mm}
\end{tabular}
\end{center}
  {\newpage}
}

\def\makeacceptance{ 
\newpage\thispagestyle{empty}
The Senior Project entitled
  {\large\bf\begin{center}\@title\end{center}}
\noindent 
developed by
  {\begin{center}{\large\bf\@firststudent}\\\@degree~in \@firststudentdeg\end{center}}
  {\begin{center}{\large\bf\@secondstudent}\\\@degree~in \@secondstudentdeg\end{center}}
\noindent
and submitted in partial fulfillment of the requirements
of their respective \@degree\ degrees
is hereby approved and accepted by the 
Department of Computer Science,
College of Computer Studies,
Ateneo de Naga University.

\begin{center}
\begin{tabular}{ccc}
\\\\\\
\textbf{\@deptchair} 
 & ~~~~~~~~~~~~~~~~~~ & \textbf{\@dean}\\
Chair, Department of Computer Science
 & & Dean, College of Computer Studies\\
Date signed: \rule{3cm}{0.15mm}
 & & Date signed: \rule{3cm}{0.15mm}
\end{tabular}
\end{center}
  {\newpage}
}

\def\makedeclaration{ 
\newpage\thispagestyle{empty}
{\LARGE\bf \begin{center}Declaration of Original Work\end{center}}
We declare that the Senior Project entitled
  {\large\bf\begin{center}\@title\end{center}}
\noindent
which we submitted to the faculty of the
  {\large\bf\begin{center}Department of Computer Science, Ateneo de Naga University\end{center}}
\noindent
is our own work. To the best of our knowledge, it does not contain materials published or written by another person, except where due citation and acknowledgement is made in our senior project documentation. The contributions of other people whom we worked with to complete this senior project are explicitly cited and acknowledged in our senior project documentation.

We also declare that the intellectual content of this senior project is the product of our own work. We conceptualized, designed, encoded, and debugged the source code of the core programs in our senior project. The source code of third party APIs and library functions used in our program are explicitly cited and acknowledged in our senior project documentation. Also duly acknowledged are the assistance of others in minor details of editing and reproduction of the documentation.

In our honor, we declare that we did not pass off as our own the work done by another person. We are the only persons who encoded the source code of our software. We understand that we may get a failing mark if the source code of our program is in fact the work of another person.

\smallskip
\begin{center}
    \begin{tabular}{ccc}
        \textbf{\@firststudent} & & \textbf{\@secondstudent}\\
        4 - \@degree\ in \@firststudentdeg & & 4 - \@degree\ in \@secondstudentdeg\\
    \end{tabular}
\end{center}

\bigskip
\bigskip
\noindent
This declaration is witnessed by:
\\\\
\textbf{\@projadvisor}\\
Project Advisor
  {\newpage}
}

%in\smallskip\\
%Blacksburg, Virginia\par\vfill

% Make Abstract page
\renewenvironment{abstract}{%
  \newpage\thispagestyle{empty}
  {{\Large\bf\begin{center}\@title\end{center}}
   \begin{center}by\\
   \@firststudent \hspace{0.1cm}and \@secondstudent \par%
   \@abstractchairman\bigskip\\{\large\bf ABSTRACT}\end{center}}}
  {\newpage}

% Make Executive Summary page
\newenvironment{execsummary}{%
  \newpage\thispagestyle{empty}
  {{\Large\bf\begin{center}\@title\end{center}}
   \begin{center}by\\
   \@firststudent, \@secondstudent, \@thirdstudent, and \@fourthstudent \par%
   \@abstractchairman\bigskip\\{\large\bf EXECUTIVE SUMMARY}\end{center}}}
  {\newpage}

% Make Dedications Page
\newenvironment{dedication}{\thispagestyle{plain}\vspace*{\fill}\begin{center}%
  \begin{minipage}{5in}}%
  {\end{minipage}\end{center}\vspace*{\fill}\newpage}

% Make Acknowledgements Page
\newenvironment{acknowledgements}{\chapter*{ACKNOWLEDGEMENTS}
  \thispagestyle{plain}}{\newpage}

% Make Vita Page
\newenvironment{vita}{\chapter*{VITA}}{\newpage}

\def\labelenumi{\arabic{enumi}.}
\def\theenumi{\arabic{enumi}}
\def\labelenumii{(\arabic{enumii})}
\def\theenumii{\arabic{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{\arabaic{enumiii}.}
\def\theenumiii{\arabaic{enumiii}}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{\arabic{enumiv}.}
\def\theenumiv{\arabic{enumiv}}
\def\p@enumiv{\p@enumiii\theenumiii}

\def\labelitemi{$\bullet$}
\def\labelitemii{\bf --}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}

\def\verse{\let\\=\@centercr
 \list{}{\itemsep\z@ \itemindent -1.5em\listparindent \itemindent
 \rightmargin\leftmargin\advance\leftmargin 1.5em}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\listparindent 1.5em
 \itemindent\listparindent
 \rightmargin\leftmargin\parsep 0pt plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\list{}{\rightmargin\leftmargin}\item[]}
\let\endquote=\endlist

\def\descriptionlabel#1{\hspace\labelsep \bf #1}
\def\description{\list{}{\labelwidth\z@ \itemindent-\leftmargin
 \let\makelabel\descriptionlabel}}
\let\enddescription\endlist
\newdimen\descriptionmargin
\descriptionmargin=3em

\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \else \newpage \fi \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

\arraycolsep 5pt \tabcolsep 6pt \arrayrulewidth .4pt \doublerulesep 2pt
\tabbingsep \labelsep
\skip\@mpfootins = \skip\footins
\fboxsep = 3pt \fboxrule = .4pt

\def\thepart {\Roman{part}}
\def\thechapter {\arabic{chapter}}
\def\thesection {\thechapter.\arabic{section}}
\def\thesubsection {\thesection.\arabic{subsection}}
\def\thesubsubsection {\thesubsection .\arabic{subsubsection}}
\def\theparagraph {\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph {\theparagraph.\arabic{subparagraph}}
\def\@chapapp{Chapter}

\def\@pnumwidth{1.55em}
\def\@tocrmarg {2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{2}

\def\beginbody{\clearpage\pagenumbering{arabic}\pagestyle{headings}}
\def\tableofcontents{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\chapter*{TABLE OF CONTENTS\@mkboth{CONTENTS}{CONTENTS}}
 \@starttoc{toc}\if@restonecol\twocolumn\fi}
\def\l@part#1#2{\addpenalty{-\@highpenalty}
 \addvspace{2.25em plus 1pt} \begingroup
 \@tempdima 3em \parindent \z@ \rightskip \@pnumwidth \parfillskip
-\@pnumwidth
 {\large \bf \leavevmode #1\hfil \hbox to\@pnumwidth{\hss #2}}\par
 \nobreak \endgroup}

\def\l@chapter#1#2{\pagebreak[3]
 \vskip 1.0em plus 1pt \@tempdima 1.5em \begingroup
 \parindent \z@ \rightskip \@pnumwidth
 \parfillskip -\@pnumwidth
 \bf \leavevmode #1\hfil \hbox to\@pnumwidth{\hss #2}\par
 \endgroup}
\def\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\def\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\def\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\def\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\def\l@subparagraph{\@dottedtocline{5}{12em}{6em}}
\def\listoffigures{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\chapter*{LIST OF FIGURES\@mkboth
 {LIST OF FIGURES}{LIST OF FIGURES}}\@starttoc{lof}\if@restonecol
 \twocolumn\fi}
\def\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\def\listoftables{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
 \fi\chapter*{LIST OF TABLES\@mkboth
 {LIST OF TABLES}{LIST OF TABLES}}\@starttoc{lot}\if@restonecol
 \twocolumn\fi}
\let\l@table\l@figure


\def\singlespace{\def\baselinestretch{1}\@normalsize}
\def\endsinglespace{}


\def\thebibliography#1{\chapter*{REFERENCES\@mkboth
 {REFERENCES}{REFERENCES}}\singlespace\list
 {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep
 \usecounter{enumi}}
 \def\newblock{\hskip .11em plus .33em minus .07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}
\def\endthebibliography{\endlist\endsinglespace}

\newif\if@restonecol
\def\theindex{\@restonecoltrue\if@twocolumn\@restonecolfalse\fi
\columnseprule \z@
\columnsep 35pt\twocolumn[\@makeschapterhead{Index}]
 \@mkboth{INDEX}{INDEX}\thispagestyle{plain}\parindent\z@
 \parskip\z@ plus .3pt\relax\let\item\@idxitem}
\def\@idxitem{\par\hangindent 40pt}
\def\subitem{\par\hangindent 40pt \hspace*{20pt}}
\def\subsubitem{\par\hangindent 40pt \hspace*{30pt}}
\def\endtheindex{\if@restonecol\onecolumn\else\clearpage\fi}
\def\indexspace{\par \vskip 10pt plus 5pt minus 3pt\relax}

\def\footnoterule{\kern-3\p@
 \hrule width \textwidth
 \kern 2.6\p@}
\@addtoreset{footnote}{chapter}
\long\def\@makefntext#1{\parindent 1em\noindent
 \hbox to 1.8em{\hss$^{\@thefnmark}$}#1}

\setcounter{topnumber}{2}
\def\topfraction{.7}
\setcounter{bottomnumber}{1}
\def\bottomfraction{.3}
\setcounter{totalnumber}{3}
\def\textfraction{.2}
\def\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.5}

\long\def\@makecaption#1#2{
 \vskip 10pt
 \setbox\@tempboxa\hbox{#1: #2}
 \ifdim \wd\@tempboxa >\hsize #1: #2\par \else \hbox
to\hsize{\hfil\box\@tempboxa\hfil}
 \fi}


\def\thefigure{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{Figure \thefigure}
\def\figure{\@float{figure}}
\let\endfigure\end@float
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}
\def\thetable{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{Table \thetable}
\def\table{\@float{table}}
\let\endtable\end@float
\@namedef{table*}{\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}

\mark{{}{}}

\if@twoside \def\ps@headings{\let\@mkboth\markboth
%%\def\@oddfoot{\hbox{}\hfil \rm \thepage \hfil}
%%\def\@evenfoot{\hbox{}\hfil \rm \thepage \hfil}
\let\@oddfoot\@empty \let\@evenfoot\@empty
\def\@oddhead{{\sl \rightmark} \hfil \thepage}
\def\@evenhead{\thepage \hfil \sl \leftmark}
%\def\chaptermark##1{\markboth {\uppercase{\ifnum \c@secnumdepth >\m@ne
% \@chapapp\ \thechapter. \ \fi ##1}}{}}\def\sectionmark##1{\markright
%{\uppercase{\ifnum \c@secnumdepth >\z@
% \thesection. \ \fi ##1}}}}
\def\chaptermark##1{\markboth {{\ifnum \c@secnumdepth >\m@ne
 \@chapapp\ \thechapter. \ \fi ##1}\hfil\@studentsheader}{}}
\def\sectionmark##1{\markright
{\@studentsheader\hfil{\ifnum \c@secnumdepth >\z@
 \thesection. \ \fi ##1}}}}
\else \def\ps@headings{\let\@mkboth\markboth
%%
%% These two commands changed as an experiment for Ben.
%%
%%\def\@oddfoot{\hbox{}\hfil \rm \thepage \hfil}
%%\def\@evenfoot{\@oddfoot}
\let\@oddfoot\@empty \let\@evenfoot\@empty
\def\@oddhead{{\sl \rightmark} \hfil \thepage}
\def\@evenhead{\thepage \hfil \sl \leftmark}
%%\def\chaptermark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\m@ne
%% \@chapapp\ \thechapter. \ \fi ##1}}}}
\def\chaptermark##1{\markright {{\small \@studentsheader\hfil{\ifnum \c@secnumdepth >\m@ne
 \@chapapp\ \thechapter. \ \fi ##1}}}}}
\fi
\def\ps@myheadings{\let\@mkboth\@gobbletwo
\def\@oddfoot{\hbox{} \hfil \rm \thepage \hfil}
\def\@oddhead{\sl\rightmark \hfil}
\def\@evenfoot{\hbox{}\hfil \rm \thepage \hfil}
\def\@evenhead{\hbox{}\hfil \sl \leftmark}
\def\sectionmark##1{}\def\subsectionmark##1{}}

\def\today{\ifcase\month\or
 January\or February\or March\or April\or May\or June\or
 July\or August\or September\or October\or November\or December\fi
 \space\number\day, \number\year}
\@addtoreset{equation}{chapter}
\def\theequation{\thechapter.\arabic{equation}}

\ps@plain \onecolumn \if@twoside\else\raggedbottom\fi
%% Move this to the beginbody command.
%%\pagestyle{headings}
\pagenumbering{arabic}
