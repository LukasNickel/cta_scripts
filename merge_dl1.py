import click
import tables
from pathlib import Path
import logging
import glob


@click.command()
@click.argument('input_pattern', type=str)
@click.argument('output_path', type=click.Path(dir_okay=False))
@click.option('--overwrite', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
def main(input_pattern, output_path, overwrite, verbose):
    log = logging.getLogger()
    logging.basicConfig()
    log.setLevel('DEBUG' if verbose else 'INFO')

    if Path(output_path).is_file():
        log.info('Output file exists.')
        if overwrite:
            log.info('Output file will be deleted.')
            Path(output_path).unlink()
        else:
            raise Exception(
                'Use --overwrite if you want to overwrite the old file '
                'or specify another filename'
            )

    # ctapipe default filters for dl1 files
    filters = tables.Filters(
        complevel=5,  # compression medium, tradeoff between speed and compression
        complib="blosc:zstd",  # use modern zstd algorithm
        fletcher32=True,  # add checksums to data chunks
    )

    input_files = glob.glob(input_pattern)
    with tables.open_file(output_path, 'a') as out_:
        log.info('Creating output_file')
        for file_name in input_files:
            log.info(f'Adding file {file_name}')
            with tables.open_file(file_name) as in_:
                for node in in_:
                    if isinstance(node, tables.Table):
                        log.debug(
                            f'Copying table {node.name} from group '
                            f'{node._v_parent._v_pathname}'
                        )

                        # create table or append to existing table
                        exists = (node._v_pathname in out_)
                        if not exists:
                            out_.create_table(
                                where=node._v_parent._v_pathname,
                                name=node.name,
                                description=node.description,
                                obj=node.read(),
                                createparents=True,
                                filters=filters,
                            )
                        else:
                            if node._v_pathname.startswith('/configuration'):
                                log.debug(
                                    f'Skipping table {node._v_pathname} '
                                    'because this information is duplicated'
                                )
                                continue
                            existing = out_.root[node._v_pathname]
                            try:
                                existing.append(node.read())
                            except Exception as e:
                                # usually this is about slightly different dtypes
                                # float32 vs float64 etc
                                log.debug(
                                    f'Cant append node {node._v_pathname} '
                                    f'from file {file_name}. Error message: {e}'
                                )
                                try:
                                    # try to cast the new table to the existing tables dtypes
                                    log.warning(
                                        f'Converting dtypes of table {node._v_pathname} '
                                        'to append to the existing table'
                                    )
                                    existing.append(
                                        node.read().astype(
                                            list(existing.description._v_dtypes.items()),
                                        )
                                    )
                                except Exception as e:
                                    log.error(
                                        f'Failed to append table {node._v_pathname}',
                                        e
                                    )


if __name__ == '__main__':
    main()
