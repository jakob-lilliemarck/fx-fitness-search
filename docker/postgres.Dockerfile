FROM postgres:17-alpine
COPY ./docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/
