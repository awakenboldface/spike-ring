(ns spike-ring.server
  (:require [macchiato.server :as server]
            [macchiato.middleware.defaults :refer [wrap-defaults api-defaults site-defaults]]
            [macchiato.util.response :as r]
            [mount.core :as mount :refer-macros [defstate]]
            [cljs.nodejs :as nodejs]))

(nodejs/enable-util-print!)

(def en-inflectors
  (nodejs/require "en-inflectors"))

(def inflectors
  (.-Inflectors en-inflectors))

(defn inflect-verb
  [verb]
  (-> verb
      inflectors.
      .toPresent))

(defn pluralize
  [noun]
  (-> noun
      inflectors.
      .toPlural))

(def http-shutdown
  (nodejs/require "http-shutdown"))

(.extend http-shutdown)

(defn home [req res raise]
  ;(pr (map (fn [token]
  ;           {:verb (apply get {"be" "are"} (repeat 2 (inflect-verb token)))
  ;            :noun (pluralize token)})
  ;         (:tokens (:params req))))
  (-> [{:hello "great"} {:hello "great"}]
      clj->js
      JSON.stringify
      r/ok
      (r/content-type "application/json")
      res))

(defonce server-atom
         (atom nil))

(defn start-server
  []
  (reset! server-atom (.withShutdown (server/start {:handler (wrap-defaults home api-defaults)
                                                    :port    3000}))))
(defn stop-server
  []
  (.shutdown @server-atom))

(defstate server-state
          :start (start-server)
          :stop (stop-server))


(.on js/process "uncaughtException" #(js/console.error %))

(set! *main-cli-fn* mount/start)

(defn -main
  [& more]
  (mount/start)
  (println "hello world"))

(set! *main-cli-fn* -main)
