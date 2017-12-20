(ns spike-ring.server
  (:require [immutant.web :as web]
            [ring.middleware.defaults :refer :all]))

(defn app [request]
  {:status 200
   :body   "Hello world!"})

;(web/run (wrap-defaults app api-defaults))
(def start
  (partial web/run app {:host "0.0.0.0"}))

(def stop
  (partial web/stop))
