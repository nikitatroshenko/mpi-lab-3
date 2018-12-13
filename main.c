#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

struct task_specification {
	int rows_cnt;
	int cols_cnt;
};

struct configuration {
	int strip_len;        /* number of rows in a strip */
	int block_size;        /* number of columns in a block */
};

void compute_routine(
	int world_size,
	int world_rank,
	int *data,
	const struct task_specification *spec,
	const struct configuration *config)
{
	int *recvbuf = calloc(config->block_size, sizeof *recvbuf);
	MPI_Request send_request;
	MPI_Request recv_request;
	int source = (world_rank == 0) ? 0 : world_rank - 1;
	int destination = (world_rank + 1) % world_size;
	int block_size = (spec->cols_cnt >= config->block_size) ? config->block_size : spec->cols_cnt;

	MPI_Irecv(recvbuf, block_size, MPI_INT, source, 0, MPI_COMM_WORLD, &recv_request);
	for (int j = 0; j < spec->cols_cnt; j += config->block_size) {
		block_size = (spec->cols_cnt - j >= config->block_size) ? config->block_size : spec->cols_cnt - j;

		MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
		memcpy(data + j, recvbuf, sizeof *recvbuf * block_size);

		MPI_Irecv(recvbuf,
			  block_size,
			  MPI_INT,
			  source,
			  j + block_size,
			  MPI_COMM_WORLD,
			  &recv_request);

		for (int i = 1; i <= config->strip_len; i++) {
			for (int k = 0; k < block_size; k++) {
				data[i * spec->cols_cnt + k + j] = data[(i - 1) * spec->cols_cnt + k + j] + 1;
			}
		}

		MPI_Isend(data + config->strip_len * spec->cols_cnt + j,
			  block_size,
			  MPI_INT,
			  destination,
			  j,
			  MPI_COMM_WORLD,
			  &send_request);
	}

	free(recvbuf);
}

int main(int argc, char **argv)
{
	int world_size;
	int world_rank;
	int *gathered_data = NULL;
	int *process_data;
	const int first_row[] = {1, 2, 3, 4, 5};
	const struct task_specification spec = {10, sizeof first_row / sizeof *first_row};

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const struct configuration config = {
		((spec.rows_cnt - 1) % world_size) ? spec.rows_cnt / world_size + 1 : spec.rows_cnt / world_size,
		3
	};

	process_data = calloc((config.strip_len + 1) * spec.cols_cnt, sizeof *process_data);
	if (world_rank == 0) {
		MPI_Request request;

		gathered_data = calloc((config.strip_len * world_size) * spec.cols_cnt, sizeof *gathered_data);
		memcpy(gathered_data, first_row, spec.cols_cnt * sizeof *gathered_data);
		for (int i = 0; i < spec.cols_cnt; i += config.block_size) {
			int block_size = (spec.cols_cnt - i >= config.block_size) ? config.block_size : spec.cols_cnt - i;

			MPI_Isend(gathered_data + i,
				  block_size,
				  MPI_INT,
				  0,
				  i,
				  MPI_COMM_WORLD,
				  &request);
			MPI_Request_free(&request);
		}
	}
	compute_routine(world_size, world_rank, process_data, &spec, &config);

	MPI_Gather(process_data + spec.cols_cnt,
		   config.strip_len * spec.cols_cnt,
		   MPI_INT,
		   gathered_data + spec.cols_cnt,
		   config.strip_len * spec.cols_cnt,
		   MPI_INT,
		   0,
		   MPI_COMM_WORLD);

	if (world_rank == 0) {
		for (int i = 0; i < spec.rows_cnt; i++) {
			for (int j = 0; j < spec.cols_cnt; j++) {
				printf("%5d", gathered_data[i * spec.cols_cnt + j]);
			}
			printf("\n");
		}
	}

	free(process_data);
	free(gathered_data);
	MPI_Finalize();
}