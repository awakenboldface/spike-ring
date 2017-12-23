(ns spike-ring.server
  (:require [immutant.web :as web]
            [ring.middleware.defaults :refer :all]
            [ring.middleware.json :refer [wrap-json-response]]
            [mount.core :refer [defstate]]))

(defn app
  [request]
  {:status 200
   :body   [{:hello "world!"}]})

;(web/run (wrap-defaults app api-defaults))
(def start
  (partial web/run (wrap-json-response app) {:host "0.0.0.0"}))

(def stop
  (partial web/stop))

(defstate server
          :start (start)
          :stop (stop))
