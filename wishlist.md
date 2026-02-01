Stuff to track:
- epoch MSE progression?

Refactor
- `fx-mq-jobs` Add table for "cancelled_jobs". Clear leases while moving jobs there.
- `fx-mq-jobs` Add a "retryable_job" query and a "retry_job" command. "retry_job" should be the only way to retry a job from "cancelled_jobs".

- `fx-durable-ga` Do no regenerate on hash collision - just copy the duplicate row data & evaluation (mark as such).
- `fx-durable-ga` Make use of the "cancelled_jobs" table while interrupting requests.

- `fx-fitness-search` Add commands to filter and list requests
- `fx-fitness-search` Add commands to "continue" an "interrupted" request
