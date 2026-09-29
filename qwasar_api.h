#ifndef QWASAR_API_H
#define QWASAR_API_H

#include "qwasar_http.h"
#include "qwasar_sessions.h"

/* The Session API (API.md): /v1/server and /v1/sessions.  Returns false if
 * the request's path is not the API's, so the caller can try its own. */
bool qw_api_handle(qw_store *st, conn *c, const http_req *r, const str *body);

#endif /* QWASAR_API_H */
