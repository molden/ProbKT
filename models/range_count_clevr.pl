:- use_module(library(lists)).
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9,10,11]) :: digit(X,Y).

count([],X,0).
count([X|T],X,Y):- count(T,X,Z), Y is 1+Z.
count([X1|T],X,Z):- X1\=X,count(T,X,Z).

countall(List,X,C,A) :-
    A=0,
    sort(List,List1),
    member(X,List1),
    count(List,X,C).
countall(List,X,C,A) :-
    A=1,
    sort(List,List1),
    member(X,List1),
    count(List,X,R),
    R>C.
countall(List,X,C,A) :-
    A=-1,
    sort(List,List1),
    member(X,List1),
    count(List,X,R),
    R<C.

roll([],L,L).
roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y).

countpart(List,[],[],[]).
countpart(List,[H|T],[F|L],[A|B]):- countall(List,H,F,A), countpart(List,T,L,B).

count_objects(X,L,C,S) :- roll(X,[],Result), countpart(Result,L,C,S).
