(ns spike-ring.server
  (:require [immutant.web :as web]
            [ring.middleware.defaults :refer :all]))

(defn app [request]
  {:status 200
   :body   "Hello world!"})

;(web/run (wrap-defaults app api-defaults))
(web/run app)

(web/stop)
