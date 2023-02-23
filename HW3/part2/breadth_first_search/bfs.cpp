#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
	#pragma omp parallel for 
	for (int i = 0; i < frontier->count; i++)
	{
		int node = frontier->vertices[i];
		int start_edge = g->outgoing_starts[node];
		int end_edge = (node == g->num_nodes - 1)
			        	? g->num_edges
				        : g->outgoing_starts[node + 1];
		// attempt to add all neighbors to the new frontier
		for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
		{
			int outgoing = g->outgoing_edges[neighbor];
			if(distances[outgoing]== NOT_VISITED_MARKER)
				if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
				{
					int index = __sync_fetch_and_add(&new_frontier->count, 1);
					new_frontier->vertices[index] = outgoing;
				}
		}
	}
}

void bfs_top_down(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
	frontier->vertices[frontier->count++] = ROOT_NODE_ID;
	sol->distances[ROOT_NODE_ID] = 0;
    while (frontier->count != 0)
    {
		new_frontier->count = 0;
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
		
        top_down_step(graph,frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
void bottom_up_step(
	Graph g,
	vertex_set* frontier,
	int* distances,
	int level)
{
	int index = 0;

	#pragma omp parallel for schedule(dynamic,1100)
	for (int i = 0; i < g->num_nodes; i++) {
		if (frontier->vertices[i] ==NOT_VISITED_MARKER) {
			int start_edge = g->incoming_starts[i];
			int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

			for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
				int incoming = g->incoming_edges[neighbor];

				if (frontier->vertices[incoming] == level) {
					distances[i] = distances[incoming] + 1;
					frontier->vertices[i] = level + 1;
					index++;
					break;
				}
			}
		}
	}
	frontier->count += index;
}
void bfs_bottom_up(Graph graph, solution* sol)
{
	vertex_set list1;
	vertex_set_init(&list1, graph->num_nodes);
	int level = 0;
	vertex_set* frontier = &list1;

	// initialize all nodes to NOT_VISITED
	#pragma omp parallel for
	for (int i = 0; i < graph->num_nodes; i++)
	{
		sol->distances[i] = NOT_VISITED_MARKER;
		frontier->vertices[i] = NOT_VISITED_MARKER;
	}
	frontier->vertices[frontier->count++] = level;
	sol->distances[ROOT_NODE_ID] = 0;

	while (frontier->count != 0) {
		frontier->count = 0;

#ifdef VERBOSE
		double start_time = CycleTimer::currentSeconds();
#endif

		bottom_up_step(graph, frontier, sol->distances, level);

#ifdef VERBOSE
		double end_time = CycleTimer::currentSeconds();
		printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
		level++;
	}
	
}


void bfs_hybrid(Graph graph, solution *sol)
{
	vertex_set list1;
	vertex_set list2;
	vertex_set_init(&list1, graph->num_nodes);
	vertex_set_init(&list2, graph->num_nodes);
	
	vertex_set* frontier = &list1;
	vertex_set* new_frontier = &list2;
	int level = 0;
	// initialize all nodes to NOT_VISITED
	#pragma omp parallel for
	for (int i = 0; i < graph->num_nodes; i++)
	{
		sol->distances[i] = NOT_VISITED_MARKER;
		frontier->vertices[i] = NOT_VISITED_MARKER;
	}

	// setup frontier with the root node
	frontier->vertices[frontier->count++] = ROOT_NODE_ID;
	sol->distances[ROOT_NODE_ID] = 0;
	
	int mode = 1;
	if (graph->num_nodes > 10000000)
		mode = 2;
	while (frontier->count != 0)
	{

#ifdef VERBOSE
		double start_time = CycleTimer::currentSeconds();
#endif
		if (mode==1)
		{
			new_frontier->count = 0;
			top_down_step(graph, frontier, new_frontier, sol->distances);
			vertex_set* tmp = frontier;
			frontier = new_frontier;
			new_frontier = tmp;
		}
		else
		{
			frontier->count = 0;
			bottom_up_step(graph, frontier, sol->distances, level);
			level++;
		}

#ifdef VERBOSE
		double end_time = CycleTimer::currentSeconds();
		printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
		
	}
}
